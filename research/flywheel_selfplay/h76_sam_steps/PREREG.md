# H76: SAM/ASAM versus more AdamW updates

Registered before any H76 timing, training or outcome cell (2026-10-31).
Owner instruction: `sol_v1_hwv_h76_t2300.md`. This is an exploratory,
conditional r882 diagnostic; the 91 dev contexts have been reused by earlier
waves. No sealed/holdout payload, outer test, serving promotion or live learner.

## Fixed inputs and scope

Reuse H59/H61's real-S3 H50 packet, metadata and seed-4900 five-family-fold
assignment. Checkpoint is a verified private copy of round 882:
`0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64`.
Frozen DOPE: `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`.
Source manifest file: `99a53569fcaf59f9309e69329c830ae1684adff382da3ae22e0b2645c4ebc4d4`;
manifest identity: `3eb4a62acea512f528500b043299a5968e1c9b1d1b1759bbf77a426b9fa1ecc3`.
Train all 1,219 DL-Train regression contexts (205 families); terminal evaluation
uses exactly 91 DL-Val regression contexts (19 disjoint families). The historical
fold-B families are trained here: they are not a held-out selection set. Preserve
the historical fold assignment/hash for lineage, without claiming H59 replay.
Only non-anchor adapter_10tok parameters train: exactly 29,211; verify inventory
and frozen DOPE before/after each cell and after training. Abort on mismatch.
Validate approved metadata paths, resolved paths and hashes before payload reads.

## Fixed intervention

Three paired seeds: 12, 13, 14; 20 rounds, batch 16, slate 192, constant lr 1e-4,
AdamW weight decay 1e-4, final-gradient norm clip 1.0. These explicit owner values
supersede H59's actual batch 256 and cosine 3e-4 to 3e-5; fresh CONTROL cells are
required. Reuse the actual H59 CONTROL loss: SmoothL1 beta=1 plus 0.1 KL at
softmax temperature 0.2, using gate-failure cost 2 and passing utility clipped
to [-2,4]. No scheduler, checkpoint selection, early stopping or dev tuning.

| Arm | Updates per round | Total AdamW updates | Backward passes | Radius |
|---|---:|---:|---:|---:|
| CONTROL | 2 | 40 | 40 | none |
| K8 | 8 | 160 | 160 | none |
| SAM | 2 | 40 | 80 | 0.05 |
| ASAM | 2 | 40 | 80 | 0.5 |

Uniform context sampling follows H59 code, not the separate H73 family-balanced
intervention. Round/context/candidate streams follow H59's keyed protocol with
batch 16; first two draws match across all four arms. K8 consumes six additional
steps per round; candidate RNG resets per round. Torch dropout RNG resets to the
keyed batch stream before each update, and is replayed for SAM's second pass.
All arms start from identical r882 weights and fresh AdamW states.

SAM perturbation is rho*g/||g||. ASAM uses the official author weight-only
normalizer T=abs(w)+0.01 for names containing `weight`, T=1 otherwise, with
perturbation rho*T^2*g/||T*g|| (denominator epsilon 1e-16). First gradients are
unclipped, second gradients are clipped. Restore exact original weights before
one AdamW update (one weight-decay application); restore even on second-pass
failure. Frozen parameters never enter the optimizer or perturbation.
Algorithm references: [SAM](https://github.com/google-research/sam) and
[ASAM author implementation](https://github.com/shi27feng/ASAM/blob/f156a680171db16d551c0d85cba2514fa3bff6a2/asam.py).

## Evaluation and admission

After the fixed final update only, use H61's unrevealed-only evaluator (indices
4..P-1, earliest index on exact prediction ties) for train and dev. Passing
utility is clip(-fitness,-2,4), gate failure -2. `val_reg` is the equal-family
mean of context utilities. Native H59 scores allowing revealed anchors are not
comparable. Report train_reg, val_reg and train_reg-val_reg for every seed/arm.
No dev payload is read by the timing cell. Each outcome cell opens dev once for
one final score; trainable parameters are frozen during evaluation.

For each treatment, subtract its fresh same-seed CONTROL family utility. Primary
point estimate is median across three per-seed equal-family mean gains. Bootstrap
10,000 samples of the 19 family clusters, retaining all three seed vectors
within each sampled family; fixed RNG seed 76000, same resamples for all arms.
Report percentile two-sided 95% CI, and one-sided LCB95 (5th percentile).
Admit iff LCB95 > 0.001546 AND median >= 0; exact threshold ties fail. Missing
or void cells yield no admission, never imputed zero. Three comparisons are
nominal; report multiplicity explicitly and a Bonferroni LCB diagnostic, while
retaining the owner's specified primary gate. No seed replacement or tuning.

Paired family win rate: fraction of the 19 family x 3 seed comparisons with
positive utility gain, ties count 0.5. Elo delta = 400*log10(p/(1-p)), with
family-cluster bootstrap 95% CI; report infinities as strings at p=0 or 1.
CONTROL is reference Elo 1500. Report treatment absolute Elo and delta with CI.
This compares three seeds on these dev families, not unseen-family capability.
K8/SAM/ASAM have different compute costs; report measured wall and pass counts.

## Budget, order and lifecycle

CPU prereg/runner/tests PR must merge before execution. Fairness helper PR #832
must be present in merged main before any GPU waiter is started. Use its FIFO
acquisition (30-second polls, legacy-waiter backoff >=45 seconds), a fresh ticket
per cell, no immediate retry. Queue behind existing H57/H59/H61; never stop them.
Each child must inherit and verify the parent's actual qualification.lock lease.
Release lock immediately on child completion, before CPU aggregation/writes.

Hard cap: 2,700 seconds cumulative GPU cell wall, including model/data setup,
training, evaluation, timing and failed/void cells; lock wait is separate (3h
cumulative maximum). Persist an in-progress cell before launch; an interrupted
unaccounted launch prevents resume, so the budget cannot reset. No retries.
First run one seed-12 K8 timing cell: all 20 rounds plus train-only evaluation,
discard its scores/weights. Conservative projection = elapsed * 13 * 1.25
(one timing plus 12 outcome cells with 25% reserve); abort before outcomes if
projection exceeds 2,700 seconds. Then fixed order seeds 12,13,14, each CONTROL,
K8,SAM,ASAM. Each child timeout is the remaining cumulative budget.

nice >=10, OMP/MKL threads 4, BLAS/NumExpr 1; child RSS <6 GiB, available RAM
>=16 GiB, host CPU and load <=85% of 102 cores. H59 disk guards: launch used
<94.5%, write used <95%, free >=1 GiB. No xbabe0/xbabe3 or live xbabe1 loop.
No large artifacts in Git. Retain logs, budget ledger and checkpoint digests
under ~/jope-data/h76/h76-wave. Publish small metrics/README/index after results;
failed feasibility or safety gates remain explicit unknown outcomes.
