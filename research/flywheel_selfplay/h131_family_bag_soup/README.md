# H131b — family-bootstrap adapter soup, Amendment A1 outcome

**NO_ADVANCE**, execution **COMPLETE**. [Amendment PR #1162](https://github.com/neverhuman/JopeDime/pull/1162) merged before this one authorized attempt. Original H131 remains INCOMPLETE with zero updates; its record and Amendment A1 are retained below.

Hypothesis and setup: average eight whole-family-bootstrap adapters to cancel harmful fine-tuning directions; fixed half-soup and CONTROL, seeds 97–99, fold 131097, 948 fit / 271 assessment contexts, 164 / 41 families. All inputs are DL-Train. The plan is 27 fits and 1,080 optimizer updates; soup has eight times CONTROL compute. DOPE c51a56d0 and all non-adapter state remain frozen. No member selection, new fold, tuned weight or best-seed selection.

| Arm | Median gain vs r882 | Family 95% CI | Median Elo and 95% CI | Family harm | Holm LCB | Verdict |
|---|---:|---|---|---:|---:|---|
| CONTROL | -0.027148016 | [-0.073965, 0.008560] | 1431.3 [1346.7, 1508.5] | 48.78% | -0.065683 | NO_ADVANCE |
| SOUP8 | -0.022976076 | [-0.069395, 0.013138] | 1448.8 [1366.7, 1534.0] | 41.46% | -0.069395 | NO_ADVANCE |
| SOUP8_HALF | -0.021520067 | [-0.069763, 0.014935] | 1483.0 [1404.4, 1568.7] | 31.71% | -0.074746 | NO_ADVANCE |

Completed 1,080 updates, 27 fits and nine assessment cells. Every arm had negative gain on all three seeds; no arm passed Holm or the 10% family-harm cap. Fresh r882 assessment utility 1.317514412634. All seed gains, family harm, Holm decisions and identical-vector flags are in [metrics.json](metrics.json).

Verification reconstructed all six soup/half-soup snapshots bitwise, replayed the family bootstraps and statistics, checked 27 curves and 39 original-checkpoint copies, and verified candidate custody preceded assessment. The pre-update two-copy prediction/cache regression passed.

Resources: monitor `WORKER_EXITED`, exit 0, no resource abort; wall 1,606.81 seconds, peak RSS 3.708 GiB. The two-fit-context token preflight passed with three total threads before and after construction; the maximum sampled worker count during the full attempt was four. Guards remained four total threads, 6 GiB RSS, at least 30 GiB available memory, load1 at most 86.7/102, disk below 94%, and nice at least 15. CPU only; no GPU, qualification lock or live-loop action.

Red-team assessment:

- Leakage: the final runtime read audit recorded 2,438 permitted pool/table paths and zero refusals; candidate custody was verified before assessment admission. Prior r882 DL-Train exposure still limits any generalization claim.
- Seed noise: three seeds and one fold; 24 members are not independent assessment replications. Bootstrap intervals condition on these trained models.
- Metric direction: higher H61 unrevealed-candidate utility is better; indices 0–3 are excluded. Elo is family W/T/L with r882=1500, not AUC.
- Cherry-picking and compute: every fixed member/seed/arm is required. No survivor average, assessment-tuned weight or retry. Eightfold compute remains a confound without CONTROL_320.
- Tie acceptance: median gain must exceed 0.001, every seed must improve, three-arm Holm adjusted LCB must exceed zero, and at most four of 41 families may have a strictly negative seed-mean gain. Ties never qualify; bootstrap tail areas remain exploratory.
- DOPE freeze: all 27 fits and nine candidate readouts passed their non-adapter parameter/buffer and frozen DOPE c51a56d0 before/after checks. Each fit started at the same pinned adapter identity and finished with changed parameters; only the 29,211 non-anchor adapter parameters were trainable.

Validation: 26 data-free tests and the five local records gates passed for Amendment A1. Closeout checks preserve source/prereg/fold identities, strict JSON, links and the complete diff at the publication head. These are same-agent checks, not independent human review; full CI remains deferred under the owner targeted-local policy.

Next step: Retain r882; no soup qualified. Any further experiment requires a new protocol.

Verified custody: `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H131b/astra-t1340/manifest.json`, VersionId `LoMO6QneloUYvpgQPtTQ38nTXu1oMBk7`, SHA256 `de0d027981f295fc213f880990f11c048eeac4087e5190fdca45ae3132742007`. Numerical archive `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H131b/astra-t1340/numerical-evidence.npz`, VersionId `4qFFo1yC_56yYu6rHovuf5E7DM7EboH0`, SHA256 `e8daa1a96fe0768631ce0952bcfe6bf7a9f385e16b983846951bc605a73751e4`; 142 member hashes restored and checked. Code stays in Git at `2b7049f4dd5907eb383b1debc3a807ecf9439dff`.

Recover the exact NPZ version, verify SHA256, load with `allow_pickle=False` and verify each raw uint8 member against numerical-inventory.json. Original H50 derived inputs and r882 are local inputs whose exact S3 recovery versions were not established here.

---

The following H131 closeout and Amendment A1 text is the frozen historical record from before H131b execution. Its pending-status wording is superseded by the completed H131b result above.

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


## Amendment A1 — H131b, one owner-authorized attempt

The owner explicitly authorized **one new H131b attempt** after H131 stopped with zero updates. H131 remains INCOMPLETE in its original receipts and metrics above. H131b uses exactly the same seeds 97–99, fold 131097, 27 fits / 1,080 optimizer updates, eight-member soup, fixed half soup, frozen DOPE and statistical gates. No control, member, seed or fold is selected or changed using outcomes. The scientific [PREREG.md](PREREG.md) remains byte-identical.

Before numerical imports, the worker sets OMP, MKL, OPENBLAS and NUMEXPR thread limits to one, NUMEXPR_MAX_THREADS to one and TOKENIZERS_PARALLELISM=false. Immediately after importing Torch it sets intra-op threads to two and inter-op threads to one. The pinned H46 token builder explicitly requests a four-thread native landmark workspace; H131b wraps that constructor only during token construction to force one native thread and restores the original factory even on error. All mathematical token operations and shared source hashes remain unchanged. This bounds a known explicit pool; H131 did not capture stacks proving which fifth thread triggered its stop.

Before full token construction, scoring or optimizer work, a preflight constructs tokens for the first two sorted **fit-only** context IDs and reads total OS threads from `/proc/self/status` before and after. Both must be at most four; the external monitor continues enforcing the same four-total-thread ceiling throughout the attempt, including token construction. The preflight uses no assessment context and discards its outputs; it preserves Torch RNG state. Every fit and assessment token pass uses the single-native-thread wrapper. No idle-helper exception or raised guard is taken.

Unit controls exercise successful tiny-slice construction, rejection of five threads before/after construction, actual Linux status parsing, constructor restoration on error and fresh-process startup configuration. The real tiny-token preflight runs only after this amendment merges, as part of the sole monitored attempt. Failure or any later resource abort means INCOMPLETE and STOP, with no further automatic retry. Full unchanged-r882 prediction/cache regression remains mandatory before optimizer updates.

H131b uses separate custody `/home/ubuntu/jope-rescue-20260921/h131b-family-bag-soup` and create-only S3 prefix `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H131b/astra-t1340/`. Original H131 outputs and pointers are preserved. Source stays in Git; only numerical artifacts and JSON receipts go to S3. The resource ceilings remain CPU only, nice at least 15, four total worker threads, RSS at most 6 GiB, available memory at least 30 GiB, load1 at most 86.7/102 and disk strictly below 94%. No GPU, qualification lock, live-loop or protected-data action is authorized.

Status: **amendment preregistered; H131b not run**. Next step: central merge, verify the merged source and resource state, then launch H131b once and publish its complete or incomplete closeout separately.
