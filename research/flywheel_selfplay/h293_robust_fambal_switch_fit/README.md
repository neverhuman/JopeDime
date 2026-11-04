# H293 — family-balanced robust switch fitting

**PREREGISTERED; models locked, H293 outer folds unscored.**

Hypothesis: family-balanced weights and within-context normalized rank targets improve family-median gain and harm while retaining >=80% of CLIP mean gain. Primary: FAMBAL_RANKNORM. Setup: DL-Train only, 205 families / 1,219 contexts; three model seeds 293101–293103, H287 fourfold family splits and fixed tree/veto/bag settings. Huber delta is selected inside outer training folds only.

H287 CLIP regression seed 287101 passed bit exactly over 1,219 contexts: equal-family gain 0.09355105605170722. No neural forwards or optimizer updates. All 75 fitted policies (72 outer-training policies plus 3 complete primary exports; 750 serialized booster files, including shared-veto copies) have SHA-pinned files and actual-in-memory versus disk prediction checks on their own training features. No H293 outer utility score has been opened.

H287 CLIP does not clip its ordinal fit labels. H293 explicitly changes label gains for rank normalization and Huber-score-clipped targets; HUBER is winsorized LambdaRank targets, not Huber regression loss. Historical H287 hyperparameters were selected using DL-Train and are reused here; this is not independent fresh evidence. See [PREREG.md](PREREG.md) for the full target encodings, exact owner gate and concentration definition.

Verdict pending. Next: score frozen outer models after this prereg reaches origin/v1/next, report all seven arms and slot0, retain primary exports either way. No adoption or reserved-family scoring.

Model lock SHA256: `712a1e3ea6360ffe09774220573292d411c1495a5bcbd938d91ee1b95a8a73c6`.

Exact-version S3 custody is in [metrics.json](metrics.json); source/model archive: `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H293/astra-t0125/fit-source-models.tar.gz`, VersionId `qscjKf9iKiExpjuMIwEj5aLGvRLFPUhA`; member manifest: `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H293/astra-t0125/fit-custody-full.json`, VersionId `DoEdgay1N_B0SrTOn5Lb6_hjhYJeDhTk`.

Seven focused tests passed. CPU 48,50–52/nice 19/numeric1/aggregate<=4/RSS<=4GiB/MemAvailable>=10GiB guards apply. Git is limited to this prereg, README, compact metrics and two H293 index rows; code and bulk evidence are S3-versioned. Main, fastlane and hosted CI remain held.
