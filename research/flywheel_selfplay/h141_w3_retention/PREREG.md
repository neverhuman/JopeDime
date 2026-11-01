# H141: W3 on run-bound retained TRAIN completions

Owner authority is `w3_gpu_t1815.md` and `rulings_t1900.md` R3. This
prospective GPU recipe implements H113 after completion admission. It is a
development experiment on the explicitly authorized retention cohort. No GPU
work or optimizer updates preceded this registration.

## Cohort and comparability decision

R3's xbabe1 fd41 events COPY is SHA `677960532f2004f3431eff7a2875ae2e5701790f5f9d9b47d56df5a2523b4419`.
Only round/parameter-hash/DOPE metadata was projected from cohort records;
validation scores are excluded. 1,115 exact-version stdout rotations bind 796
consecutive rounds 1704–2499 to exact-version, SHA-verified retention tree
shards. The original claim cutoff, 2026-11-01T00:11:59.007659Z, remains fixed.
No outcome-quality filter selected these rounds. R3 resumed the same H141
claim after its initial locator census deadline; it did not reset compute.

The earliest native attempt by (round, source line) per context/program is
selected, including failures; 18,992 later attempts remain excluded in the
ledger. All 31,952 first attempts explicitly completed: 31,451 passes and 501
gate failures. Every input matches its original TRAIN table SHA, S1 evaluator
`e3be9386…`, `jope.s1-two-fold/v1`, and `full_train_cv`. Postfix trees rebuild
losslessly to the recorded program SHA. Current catalog v246 and the H65
family/table joins supply the role closure. No native evaluator is run.

There are 1,213 contexts across all 205 H65 families. Six contexts have no
attempt; 37 more have exactly four distinct candidates. Thus 1,176 contexts
support four history anchors plus a new training candidate. The original
1,219 contexts and 1,404,287 frozen H50 pool slots stay in the census, with
unproven original pool outcomes unknown. **This is a retention-cohort
amendment, not an admission of the old pool's numerical signatures.**

The comparability decision permits conditional retained-cohort development
comparisons with the same candidates and history for every policy. Report
unknown counts and identities; never fill utilities or call conditional means
complete original-panel means. Raw missing utilities are unbounded. Therefore
this experiment cannot produce a complete-original-panel ADVANCE, a serving
promotion or an adoption recommendation. This limitation is fixed before any
held-family model scoring. No selected current DL-Val result enters fitting or
readout. Producer and r882 exposure to TRAIN families makes this development
evidence, not clean held-family generalization or unbiased off-policy evidence.

## Fixed models and training

**No W1 winner, comparators per W1 arms.** Rescore the r882 continuity COPY,
frozen 150D ridge and zero prior on this cohort. Include best-of-revealed and
matched random4 diagnostics. Do not reuse old r882 utility or Elo.

Each trained cell strictly reloads COPY SHA
`0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64`.
No adapter, normalization or anchor reset replaces those bytes. DOPE and its
normalizers stay frozen at
`c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`;
any tensor-hash difference aborts. r882's full TRAIN exposure is disclosed.
Use H65 fold SHA `cabcfc6f…`, seed4900, and seeds701–703. All six arms,
five folds and three seeds are retained: 90 cells, exactly 40 updates each,
3,600 planned updates. One fixed terminal EMA, no seed or endpoint selection.

History is the first four program-SHA-ordered first-attempt records. Every arm
uses the H58-B repaired context/prior interface, the real frozen 584D context,
10x32 tokens, frozen ridge and SoS priors, and history scale=max(std(F4),.1).
Gate failures use the explicit H58 history/rank cost 2; that is not a measured
passing residual. Fit families exclude the held fold. Minibatches sample 32
families uniformly, then one eligible context per draw. Each 192-wide slate
contains the four fixed anchors and 188 unrevealed rows, sampled with
replacement when the retained context is smaller. Repetition is explicit and
paired across arms; it is not 192 distinct native evaluations. All arms use the
same RNG slate plan and dropout seed at each update; receipts compare exposure
hashes. There are no native fits or new GP proposals.

The inherited adapter has 23,546 active parameters; frozen anchor and dead
parent-delta parameters are excluded. AdamW uses lr3e-4, weight decay1e-4,
cosine to3e-5 over40 updates, gradient norm cap2, and EMA decay.9. No frozen
feature statistics are refitted. Four H113 loss/gradient parity cases are exact
on CPU; a charged CPU/CUDA parity check must also pass under the GPU lock.

* a: H113 revealed-masked SmoothL1(beta1)+.1 KL(temp.2).
* b: H113 H58-D dual-temperature KL, original margins and std floor.05.
* c: only the passing scale changes to max(F_min(16,npass)−F1,.05).
* d: measured logratio SmoothL1, passing F−.9logratio SmoothL1, and gate BCE,
  equal component weights and missing masks from H113.
* d_scalar: three scalar-cost heads, each trained with a's scalar objective;
  average their losses. It has exactly the same backbone and head widths as d.
* d_shuffled: d's component targets jointly permuted within unrevealed slate
  positions by a fixed independent RNG. Revealed positions and masks stay fixed.

The d family shares a 35→32→3 SiLU head: candidate query32, r882-adapter cost,
frozen prior and scale. Final weights/bias are zero initially. d predicts
quality=adapter_cost/.9+delta0, passing residual=delta1, gate logit=delta2.
d_scalar predicts adapter_cost+delta_j in each channel. All three have 1,251
new parameters, 24,797 total, equal forward widths, steps, slates and data.
Loss operations differ by intervention; actual per-cell wall time is reported,
not asserted identical. Selection before any learning preserves the complete
r882 model's costs and choices; 48 CPU cases cover all six arms.

After 40 updates, each d-family cell fits the same fixed ridge1 affine
calibrator from its three head outputs to **original raw utility**, only on
unrevealed fit-family candidates. Features standardize on those fit rows;
families have equal weight, contexts equal weight within family, and candidates
equal weight within context. The intercept is unpenalized. No held target
selects calibration, sign, weights or endpoint. At zero updates no calibrator
has been learned and the selection score remains exactly r882's score. The
scalar and shuffled controls include identical calibration machinery. A d gain
matched by either control does not isolate decomposed supervision.

Five independent fit-only CPU pilots, seed700, use64 hash-chosen families in
each fold's fit set. They report frozen r882–ridge variance proxies/detectable
effect estimates and CPU runtime; they select no recipe or dose. Their source
and output hashes are in the manifest. GPU feasibility is bounded by the hard
wall ceiling rather than an endpoint search.

## Readout

Follow the PROPOSALS common raw metric: pass u=−fitness, otherwise u=−2.
`val_reg_inner` is its context mean; report the equal-family mean and median
separately. Primary legacy selection is raw argmin on the declared full retained
slate including anchors. Also report the unrevealed-only choice diagnostic.
U4 selects four distinct new programs before reading their outcomes and retains
the best together with the original four; n<8 remains untested. Report U0 and
U4, with matched r882-top4, ridge-top4 and random4. All policies are rescored.

The two prespecified primaries remain c−b and d−a. Average seeds within family,
10,000 paired family bootstrap draws, seed20261031, one-sided alpha.025 per
primary. The conditional gate is lower bound>+.001, median>=0, <=10% families
losing>.01, and no excess original gate-failure rate. Require the d gain to
clear scalar and shuffled controls too. A conditional result never overrides
the original-panel qualification limit above. U4 cannot rescue a failed legacy
primary. Retain every seed, fold, incomplete attempt and original-precision row.

Use the repository family-grouped Bradley–Terry implementation, fixed complete
contender panel on the explicitly observed common contexts, r882=1500,
prior.01, tie margin.0001, 400 iterations. Label the conditioning and full
context gaps beside Elo; do not transfer that scale to another panel. There is
no AUC, heldout confirmation or serving promotion.

## Resources and lifecycle

CPU nice19, BLAS/OMP1, one affinity CPU and host load<=85%. GPU only on xbabe2,
after this prereg is squash-merged, under inherited exclusive fd9 from
`flock -x ~/jope_plan/qualification.lock`. SHA-verify launch source and inputs.
Charge the complete GPU-lock wall interval, including imports, parity,
baselines, failed cells and calibration, against the same 7,200-second ceiling;
stop at7,100 seconds internally, with external timeout7,200. Original spend is
zero. Failed attempts do not reset the budget. Partial update ledgers persist.
Never access xbabe0 or write the live loop. Release the lock on exit.

Acceptance before launch: strict CPU r882 load; complete provenance/component
census; role/alias closure; H113 loss and gradient parity; full-model zero-update
choice invariance; frozen DOPE; Ruff; exact-head review and squash merge.
Publish the exact-version data/code/evidence manifest, then one closeout PR
with measured execution, complete censoring, resource receipts and verdict.
