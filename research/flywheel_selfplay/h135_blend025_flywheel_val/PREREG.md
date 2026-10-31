# H135 preregistration — fixed BLEND_025 on flywheel validation

Written before executable implementation and before any new scoring. One attempt after this preregistration and executor merge. No runner edits after merge; any error requires a prospective amendment before further outcomes. Source and input pins are in [manifest.json](manifest.json).

## Question and immutable recipe

Does `H129.blend(r882_cost, gbm_cost, 0.25)` improve the flywheel's own 91-context mean? Lambda is fixed. PURE_GBM and BLEND_050 are report-only and cannot qualify. Import the merged H129 runner unchanged; it hash-imports H124. Call H124's actual `fit_one`, feature construction, target centering, HistGradientBoostingRegressor parameters and calibration function. No copied model recipe or parameter tuning.

The full-train adaptation replaces H129's outer cross-fitting: virtual partition 0 contains every eligible DL-Train row; virtual partition 4 contains only validation prediction targets. Exactly three fits, one per seed. Capture the estimator constructed inside the imported fit function, without changing its parameters or data, to calibrate on each training context's **unsampled candidates at indices >=4**. Call imported `calibration` on those predictions, raw fitness and gates; cost is float32 `a-b*prediction`. Calibration is training-only but shares training contexts and is **not family out-of-fold calibration**. This differs from H129's nested cross-fit calibration and is disclosed before scoring. No validation labels enter fitting or calibration. H129's U4 endpoint is not used.

Seeds are 135,136,137 as explicitly assigned. Each gets 192 candidates per training context, without replacement from indices >=4, RNG `135000+seed`, and HistGB `random_state=seed`. Keep original selected-packet training order. Exclude every training family present in validation and report removals. Metadata preparation found 1,219 train / 91 validation contexts, 205 / 19 families, zero family overlap. No scientific payload has yet been scored.

**Seed audit:** H133 PR #1186 also reserves 133,134,135; only its seed-133 calibration is recorded at preparation time. This is a disclosed reservation collision, not evidence that135 was executed. Retain the explicit H135 assignment. Re-audit receipts immediately before launch and record no prior execution of135,136,137; if any has been used, stop for a prospective seed amendment before any H135 outcome. No silent seed replacement or false claim that the reservation is unique. Peers were notified in coordination records and on H133's PR.

## Exact comparison and mandatory sanity gate

Protocol `fixed-dev-cost-argmin-negative-fitness/v1`; selected packet SHA256 `981960ad628f5029f8c54552c7b1cb1d440072e7e9dcffecdea52c6a3493a0f1`. Read-only reference copy from xbabe1 `~/jope-data/flywheel-full-20261030/run-20261030/best_dev_checkpoint.json`, SHA256 `30853cad78d860870338316dfb5a4e869e46dc5c22d42fe4ab9d5f24f31c373a`, evaluation identity `e79a7715c1531ccb8c4055ab96971101cd4034ec3a71d83bca6d25b0e604ca83`. Match ordered context IDs, pool hashes and training-table hashes to the contract. The model source and cache payloads are hash-pinned by imported H124; never read its source_manifest or other excluded metadata.

Baseline uses an offline copy of r882 from `~/jope-data/h134/run-20261031T2036Z/r882-copy.pt`, SHA256 `0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64`. Imported baseline predicts every candidate using pinned cached tokens/projections, scale .4208, first four revealed candidates, and full-pool argmin. Utility is negative raw fitness when the selected gate passes, otherwise -2; **no clipping**. Exact cost ties use NumPy's first argmin. Before fitting or reading any blend, reproduce mean 1.1377942 and the full ordered reference utility vector within absolute 1e-6. A mismatch stops the entire attempt: no blend read, amendment required. Store fresh baseline utilities and use them as the paired comparator. DOPE composite `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814` must match before/after baseline; GBMs do not load or modify the neural model.

## Fixed readout and gates

Primary is median over three seeds of the 91-context mean gain. Bootstrap whole validation families with replacement, 10,000 draws, RNG135100, preserving **context-count weights** in each draw; then take the median of the three seed means. One-sided 95% LCB is the .05 quantile. No multiplicity correction is needed for the single registered primary; report-only arms cannot replace it.

All gates are required: strict median gain >.001; LCB>0; all three seed means>0; harm in at most10% of families (family seed-mean delta<-.01); positive trimmed mean after removing ten largest-absolute seed-mean context gains; and no pair of seeds with identical per-context utility vectors. Rank trim ties by ascending context ID. Report pairwise Pearson gain correlations (constant-vector correlation is null), exact identical-vector flags, signed and absolute top-ten gain shares (zero denominator ->null), -2 failure counts for each seed/arm, and all gates. Exact utility ties have zero gain.

Report median seed Elo against r882=1500, equal weight per family and equal context weights within family, exact W/T/L signs and half-point ties, symmetric .01 prior. Elo is `1500+400*log10((wins+.01)/(losses+.01))`; it is descriptive, not an admission substitute or AUC. Full utilities/choices stay in numerical evidence; metrics.json must be below100KB.

## Resource and read boundaries

CPU only, nice19, four total worker threads maximum, RSS strictly below8 decimalGB, own scratch below2GB, root filesystem below93.8% including reserved capacity, load below85% of available CPUs, available memory at least16GiB. External monitor samples every.25s and terminates only its own process group on breach. Baseline and each fit run in separate processes. To honor the total-thread ceiling across multiple BLAS libraries, worker OMP/MKL/OPENBLAS starts at1 (a stricter operational cap than the brief's requested4); baseline Torch and fit OpenMP explicitly allow up to4. No extra monitor thread inside the worker. This conservative resource reduction is preregistered, not a model-recipe change. No GPU or qualification.lock.

The specific DL-Train plus91 DL-Val packet and its pinned caches are authorized. Read neither sealed/holdout/meta-test payload nor metadata. Source_manifest is deliberately excluded. Live flywheel stays untouched; the designated reference JSON was copied read-only. Runtime audit rejects protected/live paths and data outside the explicit input allowlist, and refuses input writes. A missing or aborted stage is INCOMPLETE, never a zero result. No retries or survivor selection.

After the one attempt, put numerical arrays and JSON receipts create-only under `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H135/`. Verify exact-version recovery, hashes and sizes before deleting owned scratch. Source remains in Git. Preserve failed-attempt evidence too.

## Interpretation

r882 was selected as the maximum over about2,385 rounds on this same validation cohort. That selection favors r882 and makes the comparison conservative for the blend relative to that incumbent; it does not make this an independent generalization test. Validation is selector-exposed, the training calibration is within-context, and bootstrap uncertainty conditions on three fitted models. Passing means **CANDIDATE for live-adoption review only**, never promotion. Failure means retain r882. No live training/HWM/serving change is authorized.
