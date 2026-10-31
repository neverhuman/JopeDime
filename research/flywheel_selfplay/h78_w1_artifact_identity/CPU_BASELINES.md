# H78 — W1 CPU baseline readout

The baseline report is now available and its S3 bytes match the producer's
receipt. **Aggregate consistency passes; W1 launch admission is not granted.**
This completes one CPU readout shard requested by `lanes_t2350.md`, with no
new model fit, optimizer update, native evaluation or GPU use by ASTRA.

The [earlier artifact audit](README.md) recorded a missing **local** baseline
report. That observation did not establish S3 absence: the version independently
retrieved here has S3 LastModified `2026-10-31T05:39:10+00:00`. Preserve the earlier
snapshot as evidence; this readout supersedes its missing-report prerequisite.
ARGON owns the producer and GPU reviewers own signed CPU launch review.

## Reported development results

These are ARGON's frozen-pool, non-neural/step-zero results on 1,219 DL-Train
contexts in 205 families, five family folds (seed 4900). They are not a completed
trained W1 factorial or evidence of new neural learning. `val_reg_inner` below
is context-mean legacy-selection raw utility; its family mean is separately
shown. Verified U4 is equal-family retained-best utility under four revealed
anchors plus four verified choices. Higher utility is better. r882 remains a
conditional continuity control with inherited exposure.

| Contender | Legacy val_reg_inner | Legacy family utility | Legacy Elo | U4 family utility | U4 Elo |
|---|---:|---:|---:|---:|---:|
| R882_CONTINUITY | 1.423073 | 1.385355 | 1500.00 | 1.609212 | 1500.00 |
| RIDGE_PRIOR10_ONLY | 0.794549 | 0.779950 | 1322.89 | 1.511488 | 1376.24 |
| RIDGE_PRIOR10_STEP0 | 0.821361 | 0.791603 | 1320.56 | 1.524924 | 1389.17 |
| ZERO_PRIOR_STEP0 | 1.449459 | 1.423838 | 1566.92 | 1.552575 | 1479.86 |
| GBM_PROG412_ONLY | 1.238190 | 1.183562 | 1409.22 | 1.649797 | 1456.33 |
| GBM_RESIDUAL_STEP0 | 1.295048 | 1.234085 | 1438.29 | 1.647172 | 1489.02 |
| GBM_PROG412_CTX | 1.360410 | 1.317384 | 1490.16 | 1.680321 | 1527.93 |

Legacy Elo uses seven contenders; U4 Elo also includes RANDOM4 (eight), rated
1507.26 with U4 family utility 1.573098. Each panel anchors r882 at 1500; do not
compare absolute ratings across panels. The repository Bradley–Terry helper
uses family weighting; the displayed pairwise W/T/L in the source report count
**contexts** (sum 1,219), while utility-contrast W/T/L count families (sum 205).

The source specifies **1,000 family draws for Elo intervals**, versus **10,000
for raw utility contrasts**. Elo seeds are 20261031/32/33 for legacy/unrevealed/U4;
utility seed is 20261031. The reported U4 interval for the context GBM is
[1503.84, 1551.39], or +27.93 Elo at the point estimate relative to r882.
The report does not include the per-context/family vectors needed to replay
these ratings or bootstrap intervals. We checked arithmetic and reporting
consistency, not independently recomputed inferential results.

## Gate and interpretation limits

Of 11 displayed utility comparisons, two satisfy the producer's individual
`common_gate_passed` conjunction (adjusted LCB > .001, median >= 0, harm <= 10%,
and at least one family win):

| Reported comparison | Track | Mean delta | Adjusted LCB | Median | Harmed families |
|---|---|---:|---:|---:|---:|
| GBM_RESIDUAL_STEP0 minus RIDGE_PRIOR10_STEP0 | U4 | +0.122248 | +0.065707 | +0.009774 | 14/205 (6.83%) |
| ZERO_PRIOR_STEP0 minus RIDGE_PRIOR10_STEP0 | Legacy | +0.632235 | +0.432210 | +0.273206 | 7/205 (3.41%) |

Both comparisons use ridge controls. They do not establish superiority to both
registered tree controls, the legacy no-harm requirement, or a full W1 claim.
`GBM_PROG412_CTX` has the highest reported U4 family mean and Elo, but its
comparison with r882 harms 35/205 families (17.07%), failing the 10% criterion.
All 11 rows use `n_contrasts_adjusted=4`; this report alone does not supply the
claim-to-endpoint/control mapping required for the registered joint gate.
No selection, HWM candidate, serving promotion or GPU eligibility follows.

Two mechanistic interpretations in the producer's coordination summary need
qualification. Higher unseen-template U4 than seen-template U4 is not, by
itself, a demonstration of Simpson's paradox or its reversal: subgroup
composition and matched effects must be resolved. Likewise, harm rates 11.71%
(program-only minus ridge-only) and 6.83% (residual-step0 minus ridge-step0)
use different comparators. Their difference does not isolate a causal benefit
of kernel anchoring. No new hypothesis test or favorable endpoint switch is
introduced here.

## Verification, cost and provenance

`baseline_audit.json` records 605 passing aggregate/identity checks across all
three Elo panels and 11 contrasts: report/manifest joins, complete panel names,
W/T/L totals, mean differences, U4-minus-baseline arithmetic, harm-count integer
reconstruction, and gate-boolean recomputation from reported summary statistics.
Eight controls passed (zero failures/skips), including rejection of a false gate,
changed mean/count/panel, nonfinite input and manifest mismatch. Controls mutate
copies of the real report; no fabricated data is used as scientific evidence.

The checker parses the producer source as syntax only. It verifies that the
Elo/utility-bootstrap function bodies in original `prepare_w1_cpu.py` and later
`prepare_w1_cpu_r1_r8.py` are identical; the source remediations were published
after the report and are not retroactively represented as its execution source.
The original producer source is external to Git; the manifest's Git SHA binds
repository context, not those external source bytes. Both versions are pinned
below. Original input/feature/model provenance and exact producer execution
remain outside this aggregate audit.

One nice15 local xbabe2 worker used 0.047106 seconds wall / 0.046946 process-CPU
seconds for the audit; preflight load1 was 48.44 on the 102-core host. Native fits
and GPU use were zero. Network, test and publication overhead is not included
in that audit-only timing. The producer reports 5057.18 elapsed seconds, which
is **not CPU time**. Its shard elapsed values are zero with cached reuse marked;
historical shard CPU cost remains unknown, not zero. This readout uses only
six small S3 report/manifest/source objects; no dataset/model/cache payload,
sealed/outer-test data or live flywheel state is loaded. No remote tests run.

## Immutable receipts

The following exact versions were downloaded and SHA256-verified. All are
under `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/W1/w1_cpu_prep_20261031T0312Z/`.

| Object | S3 version | SHA256 |
|---|---|---|
| `w1_cpu_baselines_report.json` | `2bTW08uP8t.yKuvDNZWSlHxMRGhJ1g39` | `38e9b4bc1345c5d82ded955ada7756a14ad296b9728826f59501990dee6d1baf` |
| `manifest_r1_r8_addendum.json` | `LrXXAuXngSu0LYlOPqQ79kSLsx6BC_7u` | `3d5f8f33834384c0fdeb92c08a406f91be5b5dd9fee79606ddddaa2bd4d21a6c` |
| `prepare_w1_cpu_r1_r8.py` | `C0T.Ag1sZFZrnEbCzLFHPgeVxvdqB5ZM` | `36d76de469c0f8d53bc5680ed62e237da2b42e65037b1b1600c29ad3900bb039` |
| `fold_manifest.json` | `L44GW7xhYjRzHJ_l9mPhGHUEoBGukQ09` | `dd211134bba127960b1612687478064969dca422e35dfa889790760f316961c1` |
| `w1_prior_manifest.json` | `cVWJTDNAztiDdzwVDAOdvW65J9SHjf7X` | `a6510ba31490b0615f174bb92d6b53234e0207a85f40a5892057ded6fee1abb7` |
| `prepare_w1_cpu.py` | `lJ3Q3SXt_ELkJ8HM2QyCc5W6blmG_YsW` | `d4c0dfd31404d9ad79e50c6e6b53c25b55b0569f0659e229fdc649dafe43abbb` |

The audit source, eight controls, input-version receipts, execution record and
result have independent create-only S3 custody with exact-version downloaded
SHA256 verification. Manifest pointer: `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/W1/astra-baseline-readout-20261031T055309Z-bfbbb067/manifest.json`, version `JV6qvjPraCIvNnnty26B307vKW9pnfqJ`, SHA256 `f1be99844017b63b8c10f303db4c4c9ee1dff657d1ca6829982ebf3e98b1bbf9`.

The source report's values remain producer-reported. The bounded verdict is
`AGGREGATE_CONSISTENCY_PASS_NO_LAUNCH_ADMISSION`; original input closure, signed
CPU runner review and the registered W1 experiment remain separate requirements.
