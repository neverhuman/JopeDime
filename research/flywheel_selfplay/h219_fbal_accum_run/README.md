# H219 — family-balanced accumulation

**NO_ADVANCE**. Completed 45 independent fits and 1,800 matched optimizer updates, plus four non-scored smoke updates. Primary FBAL median gain -0.037872; Holm-adjusted LCB -0.065164.

| Arm / reference | Median gain | 95% CI | LCB95 | Holm p | Harm | Positive seeds |
|---|---:|---|---:|---:|---:|---:|
| CONTROL / R882 | -0.036255 | [-0.067284, -0.007972] | -0.061920 | diagnostic | 81/205 | 0/3 |
| FBAL / R882 | -0.037872 | [-0.070822, -0.008462] | -0.065164 | 1.000000 | 80/205 | 0/3 |
| FBAL_WCR / R882 | -0.026752 | [-0.053910, -0.004476] | -0.049297 | 1.000000 | 56/205 | 0/3 |
| FBAL_vs_CONTROL / CONTROL | -0.003575 | [-0.010360, +0.006232] | -0.008771 | diagnostic | 19/205 | 1/3 |
| FBAL_WCR_vs_CONTROL / CONTROL | +0.007575 | [-0.006831, +0.026477] | -0.003918 | diagnostic | 33/205 | 3/3 |

Elo (r882=1500): CONTROL 1483.403, FBAL 1482.762, FBAL_WCR 1492.718.

[PREREG.md](PREREG.md) preserves every pre-outcome H109 amendment and the
strict .001 margin, Holm2, harm and three-positive-seed decision. The same 205
families /1,219 TRAIN contexts are paired across three fresh five-fold partitions.
[fold_jaccard.json](fold_jaccard.json) reports every overlap against H109/H215
(maximum 0.223881). These are reused development families, not sealed qualification.

All cells reset independently to r882's 29,211 adapter parameters. DOPE stays
frozen. Durable proofs establish actual gradients, updates, changed parameters
and changed FIT forwards. All terminal models were required to be locked and
S3-recovered before held-out utility scoring. No checkpoint search or B selector.
Native/static TRAIN parity is recorded in [metrics.json](metrics.json).
Native91 was NOT_RUN: H109's TRAIN-only boundary excludes DL-Val payloads and
outcomes. No U4 measurement or serving promotion.

Preregistration [#1831](https://github.com/neverhuman/JopeDime/pull/1831) merged
before launch. Executed source: `ec93c316918fd8098c44769ab2f1a299660921d0`. All work used the corrected
xbabe2 lane; xbabe3 was left untouched. Attempt began 2026-11-02T11:31:53Z and ended
2026-11-02T11:38:43.395835+00:00; supervisor wall time 410.213s includes
setup, smoke and worker finalization. Numeric1/nice19, FIFO GPU0, RAM20GiB/free200GB
guards and the fixed90-minute cap applied throughout. No retry or budget reset.

[manifest.json](manifest.json) retains historical execution-source pins and adds
closeout custody; [provenance.json](provenance.json) records real-S3 data lineage.
S3 manifest: `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261102/H219/astra-t0400/closeout/manifest.json`; VersionId `_QLxtXZJZ.jEyi5W2mpOOa2Qbg8gva6b`;
SHA256 `e38bc9a26c8bf749c4ca2e86a31effd099437e1de825e442fc43af705c39ad40`. Every worker object and runtime archive member
was recovered by exact version and SHA256. Source/docs are small; bulk evidence
lives in S3. The closeout adds no training, inference or statistical replay.
