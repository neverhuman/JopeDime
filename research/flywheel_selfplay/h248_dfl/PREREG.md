# H248 — decision-focused expected-regret continuation

Owner: rd_astra [ASTRA]. Authority: `~/nudges/hw_t1035_astra_h248.md`.
Offline read-only r882 COPY; no serving promotion. Primary DFL_MIX, secondary DFL,
exact H170 CONTROL. Both treatment hypotheses enter Holm m=2. Native91 must match
1.1377941898164525 exactly after this prereg merges, before smoke or primary updates.

## Hypothesis and fixed losses

Expected regret differentiates the utility consequences of the softmin choice,
which may generate better candidates than the inherited surrogate alone.
Scores are costs: lower is chosen. On each inherited 24-program minibatch,
`p = softmax(-score / T)` and `L_DFL(c) = sum_j p_j * (cost_j - min_k cost_k)`.
Real valid costs are native `fitness`, without the CTRL target clamp. A failed gate
has native utility −2 (cost +2); for DFL only, replace its cost by the **full context
pool's worst valid native cost +1**. This replaces the failure sentinel; it is not
an additional clamp on valid labels. Count replacements per batch and per FIT pool.
A pool without a finite valid label fails closed. The regret minimum spans the
24 candidates used for that training step. Labels and regret targets are detached.
No clipping changes utility evaluation, selection, CONTROL or the KL targets.

- CONTROL: exact H146/H170 loss: mean smooth-L1 to valid fitness clamped [-4,2],
  invalid cost +2, plus ListNet KL weight 0.1 at temperature 0.2.
- DFL_MIX (primary): exact CONTROL loss + **0.5 × L_DFL**.
- DFL (secondary): **L_DFL + the exact CTRL ListNet KL term** (weight 0.1,
  temperature 0.2 and original CTRL targets/reduction).

DFL uses equal-family weighting: each sampled context is weighted
`N_FIT / (123 * n_contexts_in_its_FIT_family)` and the minibatch mean is taken.
This is the unbiased equal-family objective under the inherited uniform-context
sampling; weights are not normalized within a minibatch. CONTROL and the CTRL KL
term keep their exact original reductions. Thus CONTROL remains the exact H170
comparator; the treatment DFL component is family-weighted.

## Temperature calibration before prereg, with no assessment scoring

[temperatures.json](temperatures.json) freezes five temperatures, one per outer
fold's optimization FIT subset (same fold partition for all three seeds).
For each admitted TRAIN context, frozen r882 scores its full candidate pool with
the native first-four-index reveal convention. Take float64 population standard
deviation (`ddof=0`) within each context; T is the median of those spreads over
that cell's **123 FIT families only**. FIT ID and fold manifest hashes are pinned.
The union of these FIT subsets is the admitted TRAIN panel. No SELECT/ASSESS
utility comparison or DL-Val scoring informs T. Forward passes use FIT labels
only for the inherited four revealed observations. Calibration makes zero updates;
it does not select a temperature by utility. A nonpositive/nonfinite T fails closed.
The calibration source, raw spreads, guard/access receipts and checkpoint hashes
are retained in custody. T is never recalibrated after outcomes.

## Fixed continuation and selection

H247/H232 machinery, with immutable Git-source imports and H239 A1 guards:
47 adapter tensors / 29,211 coordinates; DOPE `c51a56d0`, anchor and every
non-adapter tensor remain frozen. All arms use AdamW lr 1e-4, betas (0.9,0.999),
weight decay 1e-4, global gradient clip 1.0, 40 updates from r882, and identical
H156 context/candidate order: 8 contexts × 24 programs (top-fitness four reveals
plus 20 sampled candidates). No dropout or optimizer-geometry changes.
[schedules.json](schedules.json) freezes all 40 updates.

Fold seed **248140** partitions 205 TRAIN families into five groups of 41.
Seeds **248101–248103** drive batch streams and bootstraps. For outer A, SELECT B
is the next fold cyclically; optimization FIT is the other three folds (123
families). [fold_manifest.json](fold_manifest.json) fixes that partition.
SELECT never enters optimization or temperature calibration; ASSESS never enters
optimization, calibration or selection for its cell. All has_meta_test rows are
excluded, with no sealed/holdout/live-loop input access.

Snapshots at steps 0/10/20/30/40. SELECT chooses earliest best equal-family mean;
accept only gain >0.001 (epsilon 1e-12) and family-bootstrap LCB95 >0, otherwise
r882. All three arm decisions for each cell are fsynced and directory-fsynced
before any trained-policy ASSESS or DL-Val scoring for that cell. Selected, best
unconstrained and step-40 policies are scored once per cell/panel, deduplicating
identical requested step numbers; r882 outputs are reused. Outer folds are
cross-fit estimates, not independent replications. No human reads partial efficacy
results to alter later cells.

After Native91, a bounded smoke makes one update per arm on the first FIT batch,
verifies finite gradients, changed adapter parameters, frozen DOPE, and exact
r882 restoration. Those **3 smoke updates** are separate from the primary
**15 cells × 3 arms × 40 = 1,800 updates**. One guarded primary process, no rerolls,
process restarts or deadline resets. A failed smoke stops the study.

## Statistics and verdict

Primary: exposed DL-Val91 (19 families), each seed averaging utilities of its five
selected cell policies. Secondary: 1,219-context TRAIN strict OOF (205 families).
Report per-seed equal-family gains, median gain, paired family-bootstrap CI95
(10,000 draws, seed 248150), one-sided p and Holm m=2 across DFL_MIX/DFL vs r882.
ADVANCE requires median >+0.002, LCB95 >0, adjusted p<0.05 and ≥2/3 positive seeds;
ties mean NO_ADVANCE. Report primary and secondary-arm verdicts separately; the
wave verdict is ADVANCE if either prespecified Holm-adjusted contrast passes.
DFL-minus-CONTROL and DFL_MIX-minus-CONTROL are descriptive additional contrasts.

Report strict, unconstrained SELECT-best and **unconstrained step-40** panels,
step-40 L2 drift over all adapter coordinates, harm families (loss >1e-4), and
canonical Elo (r882=1500, 400 MM iterations, prior0.01, 1,000 family bootstrap draws).
Compare DL-Val context means descriptively to historical plateau1.1389572; TRAIN
OOF is a different panel. Shared fallbacks/identical predictions are disclosed.
Family bootstrap conditions on these fixed candidates and the exposed development
panel; it does not establish untouched-test generalization.

Context #27 (zero-based), `43022ae6a8bc83e9e25ea4a2882389152856226f5b2bdd4668db07f2f510ae91`,
family `fam-9f67f1449b77152cbf20a6c82926f90496dacd7f05a222376edf0fdfb45f667f`,
remains in every primary panel and is reported separately for strict/step40.
No AUC, speed verdict, post-outcome tuning or serving promotion.

## Resources, deadlines and custody

xbabe2 CPU48–51, nice19, one numeric thread (within authorized ≤4), aggregate
threads ≤4, RSS <4 GiB, free disk ≥300 GB. Admission RAM≥30 GiB/load5≤100;
cooperative pause RAM<20 GiB or load5>108, resume at RAM≥25 GiB sustained30s and
load5≤100. Hard stop RAM<12 GiB sustained30s or <6 GiB immediately. Only the own
worker group is signalled for hard stops; never pause/kill another job.
No GPU, xbabe0, xbabe3, live flywheel or PR#2071 access/mutation.

Prereg merge by **2026-11-03 11:45Z**, compute stop **15:00Z**, closeout **15:45Z**;
incomplete work closes honestly. Calibration uses the same limits and deadline.
Targeted tests, Ruff, JSON/schema, diff-check and records gates precede exact-head
same-agent review and owner-authorized admin squash. Full remote CI is NOT_RUN,
not a gate under the explicit assignment. Real S3 catalog/input identities are
pinned; no synthetic study data. Full evidence uses create-only exact-version
S3 custody under `experiments/rd_breakthrough_20261030/H248/astra-t1035/`.
Only compact metrics and documentation enter Git.
