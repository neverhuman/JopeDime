# H314: dev91 lineage independence audit

Owner hw_t1030_astra_h314.md, Nov4. Analysis only; no fitting, new scoring,
feature rows or label values. This preregistration must reach origin/v1/next
before metadata joins are evaluated against archived gains. We impose the stronger
order: publish first, then archive regression, metadata classification, then gain join.

## Pinned inputs and regression

H312 close153f3569cc2d930d0c2270d1a391a2ac2382a5a7, archive
s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H312/astra-20261104T093953Z/h312_evidence.tar.gz,
VersionId IWjt3KdZCfYSgh6.XeNPrKrBXQW_MNqQ,
SHA256 d8bbd501469baaeb777ce93e5d1bf49d657a7a4aa11e8f71a008f3d0df29ed15.
Exact-version GET and SHA; extract archived utility/pick arrays and metadata only,
never models. Recompute each of roots30001–30006 family-equal gains using H312's
mean(arm within family)-mean(reference within family), then mean over families;
require exact full-precision equality and baseline1.1389572403748425 exactly.
Mismatch REGRESSION_FAIL and stop. H312 utility arrays are the sole outcome input.

Catalog v246 VersionId eJuErIwj0Pgu.lAgflPcWvvRKyuwptLC, as pinned by H313
8285eab4ee5ad473c8ac8a3bd52115a520c192e2. Source bucket veox-jopedime-datasets-use2.
Pin every required full-source-registry and split-registry metadata object to its
observed VersionId before GET; preserve SHA, source key and field provenance.
Use pinned selected.json metadata to map19dev and205TRAINfamilies to datasets.
No dataset payloads or row arrays. Dataset manifest/schema JSON may be used only
as referenced metadata, never following blob/CSV/Parquet/data pointers.

## Metadata and neighbor rule, fixed before outcomes

For every family retain all distinct member dataset records represented in selected
metadata (not a gain-dependent representative). Record source registry entry ID,
explicit raw/ingest lineage roots, dedupe_fingerprint, normalized column-name set
hash, n_rows, n_features, task, and missing-field markers. Match source IDs with
platform namespaces; a shared registry FILE alone is not a shared entry. Follow
explicit parent/source lineage identifiers only; no guessed filename prefixes,
coarse-family aliases or post-hoc semantic relabeling. Keep conflicting records
and source attribution instead of silently selecting the favorable value.
Column normalization: Unicode NFKC, casefold, trim, replace each run of whitespace
or non-alphanumeric separators with underscore, strip surrounding underscores;
form a sorted unique set. Use explicit named schema columns (target name may be
included if the source schema includes it), never invent names from feature count.
Names are metadata, not label values. Keep record-level column sets and row counts
paired; do not union schemas or mix a row count from another family member.

A dev family is NEIGHBORED iff ANY dev-member/TRAIN-member pair satisfies:
(a) same source registry entry OR a shared explicit lineage root;
(b) equal nonempty dedupe_fingerprint;
(c) Jaccard(normalized column sets)>=.80 AND abs(log(n_rows_d/n_rows_t))<=.10.
Positive row counts required; empty names and null/unknown IDs never match.
Report triggering clauses, witnesses, counts and SHA256(canonical JSON sorted IDs).
Missing fields are UNKNOWN, not evidence of disjointness. Confirm CLEAN only if
all potential TRAIN comparisons evaluate all three clauses and none matches.
A confirmed neighbor stays NEIGHBORED even if other fields are missing. Otherwise
incomplete comparisons yield UNKNOWN. Preserve observed matching and missingness
counts; never turn unavailable metadata into a claim of lineage independence.
Classification and its SHA must be fsynced before the gain-analysis process runs;
that classification process cannot open archived utility/gain files.

## Fixed analysis

Within NEIGHBORED and CLEAN: per-seed family-equal mean gain and median family gain;
pooled primary = median of six seed family-equal MEANS, preserving H312's estimand.
Also report median of seed family medians to remove ambiguity in “median gain”.
W/T/L tolerance1e-4, ties never wins; harm below-.01; archived invalid counts where
available. Exact one-sided mean-statistic sign flips enumerate all2^n assignments
(n<=19), shared signs across seeds; equality tolerance1e-14. Pooled statistic is
median of seed means. Empty groups are null/unmeasured, never zero.
Wild bootstrap:10000 shared Rademacher draws, numpy default_rng(314150) reset per
group on its sorted family order; centered family residuals, basic two-sided95% CI
and one-sided95% LCB (point minus .95 quantile of bootstrap deviation).
No studentization/correction. Single-family intervals are flagged degenerate.
Drop-top1/5 removes largest gains per seed; null if no families remain. Pooled is
median of seed-specific sensitivity means. Elo uses H312's pinned H302 two-arm MM,
label exactly "reference = r3175 slot0 (dev91)". No AUC or speed verdict.

For family-level association use median of six gains for each family. Similarity
is1 for observed source/lineage/fingerprint identity, otherwise maximum observed
column-name Jaccard across TRAIN records (row-ratio condition is reported separately).
Missing comparisons make this an observed lower bound, explicitly labelled;
Spearman uses available similarities, with average ranks and two-sided asymptotic
p, null if fewer than3 or constant ranks. No significance gate uses correlation.
Top10 families sorted by pooled family-median gain descending, then family ID.
Fisher one-sided greater tests top10 vs confirmed-NEIGHBORED indicator among19;
UNKNOWN is in “not confirmed neighbor”, never relabelled CLEAN. Flag incomplete
ascertainment and also give a confirmed-NEIGHBORED/CLEAN-only table if possible.
These are descriptive association diagnostics, not causal proof.

## Decision (precedence and missingness explicit)

First H310_LINEAGE_CONFOUNDED iff NEIGHBORED>=1 AND a measured pooled CLEAN
median<=+.001 OR pooled CLEAN wild one-sided95%LCB<=0, with parentheses:
NEIGHBORED>=1 AND (pooled_clean<=.001 OR pooled_clean_LCB<=0).
Otherwise H310_LINEAGE_INDEPENDENT iff CLEAN>=10 AND pooled CLEAN median>+.001
AND per-seed CLEAN wild LCB>0 on>=4/6. Else INCONCLUSIVE with reasons.
Empty or unknown CLEAN outcomes do not satisfy numerical comparisons; no inferred
zero. UNKNOWN families cannot supply the ten-family floor. Report metadata
coverage and limits even if either mechanical decision passes. All shadow only;
shared adaptive development and19families persist, no serving/breakthrough authority.

## Budget and records

xbabe2 CPU only, nice19, maththreads1, systemd MemoryMax4GiB, <=4 idle cores
(mpstat first), aggregateRSS4GiB, hostCPU<=85%, MemAvailable>=20GiB, scratch<=1GB.
Absolute compute stop2026-11-04T12:15Z, close12:30Z; no deadline reset.
Own scratch /home/ubuntu/jope-data/h314-astra. Create-only S3 archive under
experiments/rd_breakthrough_20261030/H314/astra-20261104T102434Z/, exact-version
GET/SHA/member verification then own scratch deletion. Claims/comms/index records.
CI freeze: nonforce v1/next only, no PR/main/fastlane/CI. No GPU access (including
nvidia-smi), forbidden cohorts/data/labels, live writer directories or foreign jobs.
