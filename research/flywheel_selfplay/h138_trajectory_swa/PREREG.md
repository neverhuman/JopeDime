# H138 — retained-trajectory adapter SWA/EMA, zero updates

Owner authorization: `~/nudges/h138_brief_t1715.md` and the later dispatch
`~/nudges/h138_rd_astra_t1742.md`. Preregister and centrally merge before any
candidate inference. Execute one bounded attempt on xbabe1; no implicit retry.

## Hypothesis and fixed arms

Averaging retained adapters from the flywheel's own trajectory may improve its
selected r882 adapter without further gradient steps. The fixed inputs are
rounds 782, 785, 859 and 882 from `run-20261030`. Use only separately copied,
SHA-256-pinned files under owned custody. Never write the live run directory,
signal the live flywheel, use a GPU, or acquire qualification.lock.

| Arm | Weights on rounds 782, 785, 859, 882 |
|---|---|
| R882 reference | 0, 0, 0, 1 |
| SWA4_UNIFORM | 1/4, 1/4, 1/4, 1/4 |
| SWA2_LATE | 0, 0, 1/2, 1/2 |
| EMA_0.5 | 1/6, 1/6, 1/6, 1/2 |

Average exactly the **47 named adapter parameter tensors / 29,211 parameters**:
`adapter_10tok.*` parameters excluding `adapter_10tok.anchor.*`. The model's
parameter inventory, not the state-dictionary prefix alone, defines this set.
Accumulate in float64 in the fixed round order above and cast once to each
original dtype. Every other tensor, including adapter normalization buffers,
BN/normalizer buffers, anchor and DOPE, comes from r882 unchanged. R882 itself
is a byte-identical state copy, not a recomputed weighted sum.

Abort before scoring if any of the 67 DOPE parameter/normalizer tensors differs
**byte for byte**, including signed-zero bit patterns, across the four inputs,
or if the canonical runtime DOPE composite is not
`c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`.
Verify the composite and hashes of all frozen state and adapter parameters before
and after inference. No optimizer is constructed or used; no backward/gradient
operation is permitted. Optimizer-state fields inside the input checkpoint are
not used. Total optimizer updates are exactly zero.

## Input and regression gates

`manifest.json` pins every checkpoint, the selected packet, reference contract,
imported numerical helpers, executor files and portable-input inventory. Copies
were checked against source hashes before and after copying. Pretrained Micro
assets and package versions match the prior exact H135 baseline environment.

The authorized packet contains 1,219 DL-Train regression contexts / 205 families
and 91 DL-Val regression contexts / 19 families, with zero family overlap.
Selected train+dev metadata is read and disclosed; this is not a zero-DL-Val-
metadata claim. Do not open source_manifest, sealed, holdout or meta-test metadata
or payloads. Range-selected cached features and projections use only each
stage's admitted contexts. Candidate primary inference permits only DL-Train
pool files. Dataset labels are immutable.

Before any candidate score, repeat r882 on the exact ordered 91-context contract
`fixed-dev-cost-argmin-negative-fitness/v1`, evaluation identity
`e79a7715` (full identity pinned in manifest.json). Require the per-context vector
and full-precision mean 1.1377941898164525 to match the pinned reference exactly
(zero vector and mean error); the displayed rounded 1.1377942 uses 1e-6 tolerance. A failed
regression stops the attempt before any arm score. The baseline read is the
owner-mandated exception to the later candidate-validation gate.

All stages use the imported, hash-pinned H124 prediction function unchanged:
full-pool first argmin and utility = negative fitness if its gate is valid,
otherwise −2, with no U4 clipping. No weight, seed, checkpoint or metric search.

## Primary family partitions and estimand

Use fold seeds **138181, 138182, 138183**. For each seed, permute the sorted 205
family IDs with NumPy default_rng(seed), assign index modulo 5, and evaluate all
five folds. Each partition has 41 whole families per fold, and every family is
assessed exactly once per seed. `fold_manifest.json` freezes these assignments.

**Zero-update interpretation of the requested cross-fit:** there is no fitting
step and no fold-specific model. The retained checkpoints have already been
exposed to DL-Train and validation selection. Thus these are family-grouped
five-fold evaluation partitions, not proof of unseen-family generalization.
Compute each fixed checkpoint/arm's predictions once and reuse the identical
context predictions for the three partitions. Do not manufacture seed variation
or count three copies of each family as independent evidence. Report identical
seed vectors and their Pearson correlation (null if constant).

Primary utility is equal-family: average native context utilities within each
family, then average across all 205 families in each partition. Primary gain is
the median of the three partition gains versus r882; because all partitions
cover the same fixed predictions, these three gains are identical by design.
Also report the context mean and median family gain explicitly, so these
estimands are not confused.

Bootstrap **205 whole families**, 10,000 draws, RNG seed 138100, for a two-sided
95% percentile interval of that primary gain. Do not bootstrap duplicated
partition reports as independent models. For Holm's three hypotheses use
one-sided paired family sign-flip tests: 10,000 shared random sign vectors,
RNG seed 138101, statistic mean family gain, p=(1+count(null statistic >= observed))
/10001. State the family sign-exchangeability assumption; these are conditional
comparisons of fixed retained models, not fresh-model replications.

Apply Holm step-down over the three fixed candidate arms at familywise alpha
0.05, with p-value ties ordered by arm name. An arm passes only if all hold:

- median partition gain strictly greater than +0.001;
- two-sided family-bootstrap 95% lower bound strictly above zero;
- Holm rejects its one-sided zero-gain hypothesis;
- all three partition gains strictly positive;
- at most 10% of families have a strictly negative mean gain (no tolerance);
- utilities are not identical to r882. Exact ties never count as admission.

Report harmed-family count, win/tie/loss counts, −2 failure contexts, the signed
and absolute top-10-family gain shares, seed correlations and cross-arm gain
correlations. Paired Elo uses equal-family gain signs, exact ties as half points,
r882=1500 and a fixed symmetric 0.01 prior to keep endpoint estimates finite.
Use the same family bootstrap for Elo intervals. Report Elo, never AUC.

If multiple arms pass, select the largest primary median gain; exact ties use
this fixed priority: SWA4_UNIFORM, SWA2_LATE, EMA_0.5. Selection uses no candidate
DL-Val prediction or label comparison.

## Gated secondary readout

If no primary arm passes, record NO_ADVANCE and do not score any candidate on
DL-Val. If one is selected, score **only that one arm, once**, on the fixed 91
contexts and compare with the already verified r882 baseline. No second arm,
rescue readout, alternate weight or replacement checkpoint is permitted.

Secondary utility retains the flywheel's context-weighted mean. Resample 19
whole families with replacement, 10,000 draws, RNG seed 138102; retain each
sampled family's contexts and their multiplicity in the mean. Report gain and
95% CI against 1.1377941898164525, harm, failure counts and equal-family Elo/CI.
For a fixed selected model, the median over its identical partition reports is
this single mean gain; it is not three validation replications. BREAKTHROUGH
requires the primary to have passed and secondary two-sided CI lower bound >0.
Otherwise NO_BREAKTHROUGH. Neither result authorizes serving promotion.

Checkpoint retention and r882 selection already used this validation metric;
secondary evidence remains selector-exposed. It cannot establish sealed-test
performance, and sealed/holdout/meta-test data remain unopened.

## Resources, launch and custody

CPU only on xbabe1, nice19, one numerical thread configured before interpreter
start and **at most four actual OS threads** for the worker and descendants.
Externally sample resources throughout every stage at approximately 0.25s.

- Hard abort at RSS >=6,000,000,000 bytes, actual threads >4, total owned custody
  scratch >=2,000,000,000 bytes, or available filesystem bytes <40,000,000,000.
- **The later t1742 free-space rule supersedes the earlier percentage guard**.
  Use f_bavail times fragment size on the custody filesystem, not a percentage
  adjusted by root-reserved blocks. Exactly 40 GB available passes this gate.
- Pause the owned worker process group with SIGSTOP at load1 >86.7 on 102 CPUs
  or MemAvailable <16GiB. Resume that same process with SIGCONT only after load1
  is strictly <80 and MemAvailable >=16GiB continuously for60 seconds. Reset the
  recovery timer on a boundary crossing; record every wait/pause interval.
- Retain hard guards while paused. Bound the complete attempt, including waits,
  by **two hours wall time**. At expiry mark INCOMPLETE and terminate only the
  owned worker group. No budget reset, automatic retry or outside PID signals.

Central review/merge precedes the one scored run. Verify the actual merged
executor and all remote copies, then run baseline, four primary prediction
panels, primary selection, optional single secondary panel and summary.

Preserve numerical predictions, averaged adapters, fold identities, input hashes,
read audits and resource/verification receipts under
`s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H138/astra-t1742/`.
Use create-only writes, exact-version GET and member hashes before deleting
owned scratch copies. Source stays in Git. Do not claim exact original-input S3
recovery without verified object versions. Close out README, metrics.json below
100 KB and the H138 index row through central review.
