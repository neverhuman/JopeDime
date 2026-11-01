# H193 — revealed-history invalid veto

Owner rd_astra / [ASTRA]; authority `/home/ubuntu/nudges/hwave_t0930_h193_astra.md`.
Preregistered before any H193 numerical outcome. One zero-update attempt; no adoption.

## Hypothesis and input closure

H191's coefficient selection did not pass its pooled-A gate. H193 tests a different
intervention: veto candidates whose revealed-history kernel assigns a high invalid
share, then select the original r882 minimum cost among survivors. The r882 model,
all kernel constants, raw costs, first-argmin ties and evaluator labels stay fixed.
This is a selection-policy diagnostic, not adapter training or proof of learning.

Use the inherited pinned real-S3 1,219 TRAIN contexts /205 families and explicit
91-context /19-family development baseline/readout. No sealed, holdout, meta-test,
outer-test payloads, GPU, xbabe0/3 or live-loop writes. The feature cache contains the
disclosed development-feature rows; inherited r882/feature exposure is not erased.

Only the first four hash-ordered, contractually revealed outcomes of each context may
enter its invalid score. These known history inputs remain available on A/Val; no
unrevealed outcome enters that score. **Only FIT contexts fit the threshold grid**;
only B outcomes choose among thresholds. Native TRAIN evaluation is required solely
to prove the infinity-policy identity before the treatment selection begins.

## Exact invalid-share definition

Use original float64 utility `where(gate, -fitness, -2.0) == -2.0` for each revealed
neighbour. This includes a passing fitness exactly +2, and avoids float32 rounding
of a near-2 value into an invalid indicator. The score is not calibrated probability.

Import H188 `kernel_term` without edits. It uses the frozen adapter's semantic and
normalized candidate-query similarities, learned softplus distance scales, nearest
`min(8, revealed_count)` neighbours, and reciprocal-distance weights
`0.75/(0.15+d) + 0.25/(1.50+d)`. Evaluate it with its `residual_transform` set to the
revealed invalid indicators, and again with all ones; divide the two float32 outputs.
This is the weighted share among actual revealed neighbours. The unchanged null
weight and `0.45*sigmoid(blend)` cancel up to floating-point arithmetic; require a
positive finite denominator and clamp rounding to [0,1]. No component is modified
inside r882's score. H192's native Engine/forward/batching and H188's kernel helper
are imported, not reimplemented.

For `p_invalid > theta`, veto that candidate. Equality survives. Select the first
r882 argmin over survivors; if every candidate is vetoed, return the original r882
pick. Infinity must yield exactly the original pick and utility for all 1,219 TRAIN
contexts, including ties. JSON `null` represents infinity, never nonstandard Infinity.

## Nested selection and lock

Seeds193101/193102/193103 map literally to distinct fold seeds193141/193142/193143.
Import H191/H177 cyclic 3/1/1 family folds:123 FIT /41 B /41 A, five outer folds,
exact-once B and A coverage per seed. No optimizer, backward or fitted model.

For each cell, concatenate positive invalid scores over **FIT candidates only**.
The ordered grid is `{inf, q90, q75, q50}`, using NumPy's linear quantiles. It is
candidate-weighted, not family-weighted. Empty positive support gives four infinity
entries. Choose highest equal-family B utility (context means within family, then
equal family mean). Exact ties prefer infinity when tied for best, otherwise the
first tied grid entry: q90 before q75 before q50.

Write and fsync all15 choices, FIT grids, family roles, B utility-panel hashes,
native-cost/invalid-score hashes, fold-manifest and adapter hashes into
`veto_selection_lock.json` **before any thresholded A or treatment DL-Val scoring**.
A/Val entry points refuse absent, incomplete or mutated locks. Native91 baseline,
native TRAIN identity and revealed-input feature extraction are explicit pre-lock
operations. No A treatment utility enters selection.

A cell's choice is invariant to perturbing its own A outcomes. The entire15-cell
lock cannot be invariant to changing a family's outcomes in other cells where that
family legitimately serves as B; the tests and claims distinguish these cases.

For descriptive DL-Val, each seed uses the median of its five B-selected numeric
thresholds, with infinity ordered high. Fixed-quantile readouts use the median of
the five corresponding FIT thresholds. All these policies are in the B lock before
A/Val. One deterministic policy per seed; no ensemble, full-TRAIN refit or Val tuning.

## Decision and reporting

VETO is the sole primary (Holm m=1). Q90/Q75/Q50 are descriptive only. Reuse H191's
five-column summaries and exact H177 selector/verdict objects; H191's internal label
`KXSEL` is a disclosed alias for VETO, not a kernel-scale intervention. Actual call
multiplicity is1; H177's legacy literal metadata4 is preserved separately.

ADVANCE requires imported pooled-A admission (seed-averaged family mean gain strictly
>+.001+1e-12, paired-family CI lower>0, Holm1 p<.05), plus median seed gain
>+.001+1e-12 and median CI lower>0, and the H178 receipt gate. Ties reject. No3/3
positive-seed gate or descriptive-result rescue. Flag the flywheel owner on ADVANCE;
never adopt or write serving state.

Bootstrap193150,10,000 paired family draws shared across seeds; median reduction on
seed axis0. Elo uses the imported canonical400-iteration Bradley–Terry fit, prior.01,
tie tolerance1e-4, R8821500,1,000 family draws193151. Report per-seed gains, median/CI,
Elo/CI, context utility/val_reg separately from equal-family effects, exact−2 counts,
veto/fallback counts and rates per threshold, and all15 selections/threshold grids.

**Seed variation is family reshuffling, not independent model replication.** Native
costs and invalid scores are seed-invariant. FIT quantiles and selected policies may
vary, so their utilities need not be identical. Three positive seeds are not three
independent models. Family intervals condition on the fitted thresholds/locked policies;
they do not re-fit selection inside bootstrap draws. Families recur in other roles
and were historically exposed. H178 calibrated original m4, not nested H193 m1;
its receipt does not establish a new empirical FWER guarantee. DL-Val remains
repeatedly exposed and descriptive, including any internal primary-role/p-value labels.

## Execution and custody

Native91 mean1.1377941898164525 and full per-context vector must match bit-exactly first.
DOPEc51a56d0 and adapter hashes remain unchanged before/after each evaluation.
CPU xbabe2; xbabe1 fallback only if xbabe2 disk guard trips. Nice19, math threads1,
aggregate threads<=4/RSS<6GB, one4h wall cap, outputs<500,000,000B. Stop on sustained
load>85% of online cores (three readings at least5s apart), CPU>85%, available RAM<16GiB,
user-visible disk>=93.8%, or free<40GB. No retry, threshold relaxation or budget reset.
Constructor smoke scores zero contexts and does not consume the scientific attempt.

Preserve native candidate costs, invalid-score vectors, panels, lock, hashes, audit
and resource receipts outside Git. Post-run reconstruction uses original labels and
saved costs without a model rerun. Metrics<100KB. Create-only S3 archives must recover
by exact version and every member hash before verified owned bulk reclamation.
Prereg review<=15min, targeted tests/Ruff/five records, merge, one run, result closeout
PR and lane cleanup. Full hosted CI is not implicitly claimed.
