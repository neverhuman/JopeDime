# H230 COSWR × GATED_025 stack — INCOMPLETE

The sole authorized attempt failed during setup after the unchanged r882 native91 check passed. **No HistGBM fit, adapter optimizer update, or treatment evaluation ran.** The hypothesis is untested; native gain, confidence interval, paired Elo, harm and plateau gain are unknown (`null`), not zero. No serving promotion or retry.

## Hypothesis and fixed setup

H136 COSWR and H217 GATED_025 might combine to clear the plateau margin. The frozen [preregistration](PREREG.md) specifies R882, COSWR, G025 and STACK, fresh seeds 230101–230103 and family folds 230141–230143 over 205 DL-Train families / 1,219 contexts. Maximum prior-wave best-match Jaccard was 0.1889053792; [fold inventory](fold_manifest.json) records the compared waves. Planned work was 45 unique HistGBMs and 15 adapter cells × 40 updates, with FIT-only gate selection. Both native and DL-Val endpoints must pass their preregistered margins; no endpoint was reached here.

## Attempt and observed outcome

Preregistration PR #2052 merged at execution source `0d7388b1aaf30629c673a40ab551a382cf939b22`. H224 closeout #2042 had already merged. The single attempt exited 1 during setup on 2026-11-02 at 20:24:09Z, after 352.401 supervisor wall seconds. Native91 reproduced bit exactly (mean 1.1377941898164525), then `Controller` raised `KeyError('pids is not a file in the archive')`. The TRAIN smoke was NOT_REACHED. There was no reboot interruption or resource stop.

DOPE remained `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814` before/after; optimizer updates 0, completed HistGBM fits 0, completed adapter cells 0, CUDA uninitialized, sealed/holdout/meta-test reads 0. Treatment A and DL-Val scoring did not run. Maximum process-tree RSS was 1034653696 bytes and load5 was 43.2158203125; fixed cores [2,22,67,104], nice19, absolute compute deadline 2026-11-03T03:30Z and all resource guards were retained.

## Failure audit and test gap

My H230 port omitted H136's raw-pool normalization. The pinned raw NPZ pools expose `program_sha256`; H136 decodes those bytes into internal `pids` before sampling/hashing. H230's `Controller.__init__` incorrectly requested `z["pids"]` directly from the raw container. This is a loader defect, not evidence against the stacking hypothesis.

The 22 preregistration data-free tests passed because sampler fixtures already supplied normalized `pids`; they did not exercise the raw NPZ boundary. A post-failure data-free counterexample using the exact executed controller and a raw `program_sha256` fixture reproduced the same KeyError without any real-data fit or optimizer update. H136's normalization produced the expected fixture IDs. The counterexample and original traceback are archived. The failing scientific source is preserved unchanged in this closeout; these tests do not establish operational qualification.

## Custody and disposition

[Compact metrics](metrics.json) and [source manifest](manifest.json) retain the failure, planned/actual inventory, source identity, resources and null scientific outputs. All three worker S3 objects were recovered by exact VersionId/SHA256. The complete runtime archive was recovered by exact version and every member checked. Closeout manifest:

- URI: `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261102/H230/astra-t1945/closeout/manifest.json`
- VersionId: `eNECE9gC_a3Qyz4_FLyjMNbaS3bgg7uT`
- SHA256: `e90e6efe376f5b3c218f768c7ce7cdbae55a7bbd3f62c60b50f44b6ec33a4807`

**INCOMPLETE, no efficacy verdict.** The owner authorized one attempt and no retry; this attempt is closed. A future separately authorized wave must normalize `program_sha256` into internal `pids` and cover the raw NPZ loader boundary before execution. No post-outcome retuning or additional training was performed.
