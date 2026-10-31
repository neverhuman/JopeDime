# H131 — family-bootstrap adapter soup

**INCOMPLETE: resource guard stopped the worker before training.** The attempt made **0 optimizer updates**, completed **0/27 fits** and produced **0/9 assessment cells**. Gains, Elo, confidence intervals, family harm and statistical admission remain unknown. This is not evidence for or against the soup hypothesis and is not a NO_ADVANCE result.

[Preregistration PR #1146](https://github.com/neverhuman/JopeDime/pull/1146) merged at `afb77155a8a6fac35c0b238330ff7b33cfc26d10` before execution. [PREREG.md](PREREG.md), executable source, seeds and input identities remain unchanged. No retry or serving promotion occurred.

## Hypothesis and registered setup

A uniform average of adapters trained on eight whole-family bootstrap resamples might cancel harmful family-specific fine-tuning directions. For each seed 97, 98 and 99, CONTROL was to train 40 updates; SOUP8 was to average eight 40-update members; SOUP8_HALF was to take the fixed halfway step from r882 toward that soup. Member seeds are `100*s+k`, for `k=1..8`. No member selection, tuning or survivor average was allowed. Soup requires eight times CONTROL's updates; the optional CONTROL_320 diagnostic was omitted.

The frozen fold seed 131097 assigns 948 DL-Train contexts from 164 families to fit and 271 contexts from 41 families to assessment. Resampling preserves each selected family's full context list and all duplicate occurrences. The plan trains only the 29,211 non-anchor adapter parameters, with all other parameters/buffers and DOPE `c51a56d0` frozen. All 27 fits and nine candidates must finish and have verified custody before assessment payload admission.

The registered primary comparison is against r882, using 10,000 paired family-bootstrap draws and Holm across all three arms. Admission requires median gain above 0.001, positive gain on every seed, adjusted LCB above zero and harm in no more than 10% of families. Harm counts any strictly negative seed-mean family gain, without a tolerance; at most four of 41 families may be harmed. Ties and unchanged r882 utilities cannot qualify.

## Observed stop and evidence

The monitor launched worker PID 565651 once at 18:58 UTC on 2026-10-31. At **18:58:37.312548 UTC**, it measured **five total OS threads** in the worker, exceeding the four-thread ceiling, and terminated its own process group with SIGTERM (exit -15). Status is `ABORTED_RESOURCE_LIMIT`, reason `worker_threads>4`. Elapsed wall time was 29.42 seconds; peak process-tree RSS was 3.067 GiB. At the stop, load1 was 61.53, disk used 93.800% and available memory 77.65 GiB. The other resource ceilings were not the cause.

The last flushed event was `fit_tokens`, with updates 0. The verified sequential source emits `regression_scoring` and completes a two-copy regression check before constructing any optimizer. Neither later event exists; there are no training directories, candidate soups or assessment outputs. These artifacts and source order support the zero-update count. The first r882 copy matches its pinned SHA256, and the initial adapter snapshot and non-adapter hash were saved.

The unchanged-r882 prediction/cache regression **did not finish**. SIGTERM also prevented the final read-audit receipt from being flushed. The source admission gate and recorded phase establish that assessment was not reached, but there is no complete runtime read audit to claim. Per-thread stacks were not captured, so the extra thread's origin is unknown. PyTorch intra-op 2/inter-op 1 and BLAS environment settings did not keep total process threads within four.

Fixed guards were CPU only, nice at least 15 (actual 19), at most four worker threads, RSS at most 6 GiB, available memory at least 30 GiB, load1 at most 86.7/102 and disk strictly below 94%. No GPU, qualification lock, live-loop modification or automatic retry was used.

## Red-team assessment

- **Leakage:** assessment admission was not reached. The final runtime audit is missing, as disclosed above. Any future completed comparison would still inherit r882's prior DL-Train exposure and would not establish unseen-family generalization.
- **Seed noise:** the planned three seeds and one fold provide limited replication. No seed outcome exists here; 24 planned members are not 24 assessment replications.
- **Metric direction:** H61 unrevealed-candidate utility is higher-is-better and excludes indices 0–3. Elo would compare family W/T/L with r882=1500; it is unknown here. No AUC is reported.
- **Cherry-picking:** no outcomes were available for selection, and the stopped attempt was not restarted. Eightfold compute remains a confound in the registered design.
- **Tie acceptance:** the preregistration excludes exact ties using its strict margin and every-seed positivity gates. No statistical gate was evaluated.
- **DOPE freeze:** the first copied model passed its initial identity checks. No optimizer was reached; completed-fit before/after checks and unchanged-prediction regression are unavailable. No full-wave freeze qualification is claimed.

## Validation, custody and next step

Before execution, 20 data-free tests passed with zero skips, along with static checks and all five records gates. Those tests included CONTROL schedule parity; runtime parity was never reached. Closeout verification checks the stop artifacts, original source pins, strict JSON, links, diff and records at the publication head. These are same-agent checks; full CI is deferred under the owner's targeted-local policy.

[metrics.json](metrics.json) retains null scientific metrics and the observed zero updates. [manifest.json](manifest.json) carries the merged source, immutable preregistration and exact S3 custody pointers. The numerical archive contains 12 restored and hash-verified members, including progress, monitor/resource samples, bootstrap identities, initial adapter and stop verification. It excludes source code and the duplicate original checkpoint.

- Manifest: `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H131/astra-t1235/manifest.json`; VersionId `hsJSCL38idqNRqNod72q8oGb7ntin7LX`; SHA256 `ae60da96a1176117a3cf1e6d754a648f592034b42af050ae41246a82c1a11fc5`.
- Numerical archive: `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H131/astra-t1235/numerical-evidence.npz`; VersionId `2v3nBtklziUHavFJi7TCxo0RZQ1vd98Q`; SHA256 `453c563de272c8cb974bcd0e1a9caa62788e13d390506d5ae98ef30af7ce7d28`.
- Source: Git commit `afb77155a8a6fac35c0b238330ff7b33cfc26d10`.

Retrieve the exact archive version and verify SHA256. Load the NPZ with `allow_pickle=False`; each uint8 array contains the exact bytes of its named relative-path member. Verify members against `numerical-inventory.json`. There are no complete cells or soups to replay. Exact S3 recovery versions for the original H50 derived inputs and r882 were not established by this wave.

**Next step:** preserve this incomplete attempt. A future separately authorized attempt must first demonstrate that the entire worker stays within four total OS threads through token construction. Diagnosing or changing that runtime configuration does not authorize a rerun, change the scientific protocol or reset this wave's budget. Retain r882; the soup hypothesis remains untested.
