# H176 — training data scaling versus adapter representation limits

Owner: `luna_ns_hbaseline`; assignment `~/nudges/hwave_t0400_h176_hbaseline.md`; lane exemption `~/jope_plan/DECISIONS_20261029.md:73`. This is a diagnostic only. It cannot advance a model, serving release, or live flywheel checkpoint. The immutable object of inference is whether adding more real-S3 TRAIN families improves an adapter initialized from its native random constructor, under a fixed number of training updates.

## Questions and frozen protocol

### A. Zero-update generalization gap

First pass the exact native r882 DL-Val gate: all 91 utilities must match the pinned reference bit-for-bit and the mean must equal `1.1377941898164525`. Before training, score r882 on all 1,219 TRAIN contexts (205 families) and the 91 DL-Val contexts (19 disjoint families). Report equal-family utility, each split's per-family utility distribution, and the count of contexts whose selected utility is exactly `-2.0`. This is descriptive; DL-Val never calibrates a training choice or verdict.

For every context, define utility for candidate `j` as `-fitness[j]` when its gate is true and `-2.0` otherwise. The per-slate oracle utility is the maximum of those 48 candidate utilities. Report oracle utility on both splits and the **oracle-headroom-normalized gap**:

`(r882_train_equal_family_utility - r882_dl_val_equal_family_utility) / (train_oracle_equal_family_utility - r882_train_equal_family_utility)`.

Retain the denominator and set the normalized value to null if TRAIN oracle headroom is not positive. A large positive raw gap is descriptive overfit/data-limited evidence; a gap near zero is descriptive underfit/capacity-limited evidence. Neither alone determines the H176 verdict.

### B. Family-nested learning curve

Use the pinned 1,219 real-S3 TRAIN contexts / 205 families, 91 DL-Val contexts / 19 separate families, cached frozen-DOPE features, and the five H148/H156/H170 family folds exactly. For outer fold `j`, `A=group[j]`, `B=group[(j+1) mod 5]`, and FIT is the other three 41-family groups (123 families). Only FIT families supply minibatches. B is reserved and is never used for fitting, checkpoint selection, or efficacy evaluation. Each selected `A` family is held out of its corresponding fit and is assessed once per seed in the five-fold design. If the timed pilot requires three outer folds, use `A` groups 0, 1, and 2 with their corresponding B/FIT groups; disclose the reduced 123-family A coverage. No family can cross a fold boundary.

For each `(outer fold, seed)`, construct one seeded native-random adapter initialization and one independent seeded family permutation of its 123 FIT families. Reuse that same initialization across the three nested data-size arms to pair data-size effects. The sizes are `31/123` families (labeled 25%; ceil), `62/123` (labeled 50%; ceil), and all `123/123`. The first 31 permuted families are a subset of the first 62; both are subsets of all 123. Each nested subset permutation is seeded from `(176140, seed, outer, "nested-subset-v1")`; the three seeds independently determine initialization, subset permutation, and minibatch streams. No synthetic or resampled context is added.

The treatment trains only the 47 non-anchor adapter tensors (29,211 parameters). For a fresh arm, initialize a new adapter instance through the pinned native `FlywheelBlockFormerAdapterNet` constructor with the seeded CPU RNG, and copy only those 29,211 parameter values onto a private in-memory r882 network. Do not reinitialize or update the frozen anchor, DOPE or any non-adapter state. Use new AdamW state, `lr=1e-4`, `weight_decay=1e-4`, `betas=(0.9,0.999)`, default epsilon, global gradient clipping L2=1, and the pinned H146/H156 SmoothL1 + 0.1 KL objective. Each update uses 8 contexts sampled uniformly from that arm's FIT-only context set; for each context use the four lowest raw-fitness candidates plus 20 sampled without replacement from the remainder. The `FRESH_100` and `R882_CONT_100` arms share the exact 100%-FIT batch plan. The continuation anchor starts from r882 with fresh AdamW state and the same number of updates. All arms must perform the same selected update count.

After all FIT updates, evaluate each final policy only on its corresponding untouched A families. B is unused by trained-policy evaluation; neither A nor DL-Val outcomes enter gradients, initialization, subset selection, or minibatch construction. The r882 reference panel is evaluated on the same A contexts. The zero-update gap is the only operation that reads/evaluates r882 outcomes over the full TRAIN roster, and those values never select or alter any treatment policy.

Seeds: `176101,176102,176103`; fixed family-fold seed `176140`; bootstrap seed `176141`. The pilot cell is fixed as outer fold 0, seed `176101`, `FRESH_25`. It performs four updates and records only elapsed time per update; pilot losses, gradients, predictions and utilities are not inspected for design decisions. Use the median elapsed time of pilot updates 2–4 as `seconds_per_update`. The timing-only execution rule is fixed in `analysis.execution_plan`: assume 40 updates per cell and reserve 900 seconds of the 10,800-second wall cap. Use five folds if `5*4 arms*3 seeds*40*seconds_per_update + 900 <= 10,800`; otherwise use three folds if its corresponding 40-update estimate fits. If neither fits, retain three folds and choose `floor((10,800-900)/(3*4*3*seconds_per_update))` updates for every arm and fraction. If fewer than four updates fit, mark the attempt INCOMPLETE after the pilot; do not assess an efficacy result. The pilot updates count in the first FRESH_25 cell. No outcomes are inspected until the final fold count and common update budget are locked and fsynced. The run monitor independently stops at three hours or any host guard.

## Outcomes and fixed inference

For each data fraction and the r882-continuation anchor, report A equal-family utility per seed, median over the three seed means, and a 10,000-draw paired family-bootstrap 95% percentile interval. For each bootstrap draw, resample A families with replacement shared across all fractions, arms and seeds; average contexts within family, compute each seed's equal-family mean, then take the median across seeds. Report paired family differences `50%−25%` and `100%−50%` by the same method. Fit the three fresh-init A-utility points against `log2(fraction) = (-2,-1,0)` separately per seed; report median slope per doubling and a shared-family-bootstrap 95% interval. Report each trained arm's canonical H148 Bradley–Terry MM Elo versus r882 (`r882=1500`, context pairwise win/loss/tie margin `1e-4`, equal family weighting, `.01` pseudo-win prior, 400 iterations) and 10,000 paired-family-bootstrap intervals. No AUC, speed, or serving metrics.

Verdict, based only on the fresh-init paired learning curve:

- `DATA_LIMITED` if the `50%→100%` paired gain's lower 95% bound is above zero.
- `DATA_SATURATED` if the `50%→100%` interval includes zero and either the `25%→50%` interval includes zero or the slope's upper 95% bound is below `0.001` utility per doubling.
- Otherwise `INCONCLUSIVE`.

`DATA_LIMITED` recommends planning a separate real-S3 TRAIN expansion from the catalog; it does not authorize S3 writes or access to sealed material. `DATA_SATURATED` recommends pausing optimizer-tweak waves and planning a separately preregistered representation/capacity study. No H176 result changes the live selector or promotes a checkpoint.

## Inputs, custody and guards

Use only the pinned cached frozen-DOPE features and locally restored real-S3 TRAIN/DL-Val pool closure from H170/H165/H144. Verify the exact native 91-context r882 baseline before opening TRAIN outcome pools. Verify catalog provenance for all 276 TRAIN/DL-Val dataset IDs; excluded/sealed-matching IDs and any `BeyondArena/**/sealed/` inputs are forbidden. No holdout, sealed, meta-test, serving, or live flywheel reads/writes. DOPE composite is pinned to `c51a56d0…`; check the DOPE and all non-adapter state bitwise before and after every update/evaluation. Persist per-cell FIT IDs, nested subset IDs, ordered batches, update diagnostics, initialization/state hashes and A panels outside Git.

Execution is CPU-only on `xbabe2`, `nice -n 19`, at most four aggregate process-tree threads, aggregate RSS strictly below 6 GB, host CPU at most 85%, `MemAvailable >=16 GiB`, free disk at least 40 GB and visible disk use below 95%. Stop at three wall-clock hours. A guard or parity failure is INCOMPLETE; no scientific retries or cell reuse. The model worker has no credentials; inputs are pinned local custody copies. Output custody is create-only/versioned S3 with byte-readback verification after the worker exits; Git carries only code, compact aggregate metrics, protocol, and index row. No GPU, `qualification.lock`, `xbabe0`, or `xbabe3`.

## Flow

One preregistration PR contains this protocol, runner/monitor/statistics and data-free tests for fold separation, nested subsets, A/B exclusion, timing-only budget choice, bootstrap summaries, and verdict boundaries. Obtain one exact-head `gpu_fused_redteam_codex` review (do not block beyond 15 minutes), run focused tests, Ruff, compile and all five `jope-records` verifiers. After authorized squash merge, verify merged source bytes and exact source/review binding before a single monitored attempt. Close out with a small README and metrics JSON, all raw panels and receipts in S3 custody, independent recount, and one preserved H-wave index row. Remove this lane after the closeout PR is merged and its source/evidence are independently preserved.
