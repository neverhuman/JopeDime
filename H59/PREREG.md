# H59 preregistration — Robust and context-normalized flywheel loss

**Status:** preregistered design only; no H59 outcome work has run.  
**Date:** 2026-10-30  
**Scope:** offline adapter loss ablation; CPU-only preparation, no training, GPU use, or dataset payload reads are authorized by this document.

## Question and limits

Does robustifying the pointwise loss, normalizing each context's loss scale, or combining both improve family-level selection utility over the current flywheel loss when the four arms use paired seeds and identical batches? The hypothesis is that a small number of extreme residuals, especially gate-failure targets at the `-2.0` utility floor, should exert less influence on adapter updates.

This is an incremental test from the pinned `r882` adapter, not an independent test of generalization from an unexposed initialization. H56 audit §2.1 (the contaminated-baseline finding) documents that `r882` was trained on all 1,219 `DL-Train` contexts, spanning all 205 families. Fold-B results therefore compare candidates with an in-sample-trained control and are expected to be conservative; they do not remove the initialization's exposure.

## Arms and fixed loss definitions

All four arms start from an independent byte-identical copy of the same pinned `r882` adapter for each seed. Within each seed, arms consume the same ordered contexts, candidate slates, permutations, and minibatches at each update. Random streams are paired by `(seed, round, step)`; arm-specific arithmetic must not alter sampling. No family reweighting is used: batches use the same family-balanced context sampler in all arms, with each sampled context weighted equally within its batch.

For context `c` and candidate `i`, let `p_ci` be predicted cost and `t_ci = -clip(u_ci, -2, 4)` be the current target. Let `l_ci = smooth_l1(p_ci, t_ci)` with PyTorch's default `beta=1`. Let `K_c` be the existing per-context KL term, computed exactly as in the current loss: `KL(softmax(-t_c / 0.2) || softmax(-p_c / 0.2))`. The current per-context objective is `L_c = mean_i(l_ci) + 0.1 K_c`; batch loss is the mean of `L_c` over contexts.

| Arm | Per-context objective |
| --- | --- |
| `CONTROL` | `L_c` as defined above. |
| `CLIP` | `mean_i(min(l_ci, 1.0)) + 0.1 K_c`. The fixed cap is 1.0 loss unit per candidate; the KL term remains unchanged. |
| `NORM` | `L_c / s_c`. |
| `CLIP+NORM` | `[mean_i(min(l_ci, 1.0)) + 0.1 K_c] / s_c`. |

For both normalized arms, compute `s_c = max(1.4826 * median_i(abs(t_ci - median_j(t_cj))), 0.05)` from that context's training slate targets only. The scale is detached from autograd and is recomputed per sampled training context. The floor is 0.05 target-cost units. Do not estimate it from validation or admission data. Clipping is applied to each Smooth-L1 candidate term before candidate averaging; normalization is applied afterward to the complete per-context objective, including the unchanged KL term. The CONTROL and NORM arms preserve the current un-clipped Smooth-L1 term.

## Initialization, split, and evaluation firewall

- Pin and record the exact `round-882.pt` SHA-256, source manifest identity, and family-fold assignment hash before execution. Expected DOPE SHA-256 is `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`.
- Use the H55/H57 five-fold, family-grouped `DL-Train` assignment with seed `4900`. Freeze and hash the assignment before any outcome work. Fold set A is four folds; fold set B is the remaining fold. Train only on A. No fold-B labels, targets, or scores may influence updates, stopping, checkpoint selection, or arm selection.
- After training, make exactly one admission evaluation on untouched B for each final seed/arm candidate. Score the pinned `r882` copy on those same B contexts as the paired control. Note the conservative bias from `r882` having seen B during its earlier training.
- A fold-clean initialization sidecar in the style of H58 is **needed before making any broader claim that the effect is independent of r882's exposure**. It is outside H59's four-arm primary comparison and is deferred; H59 results must be described as conditional on the exposed r882 initialization. Do not silently add sidecar cells to H59 or use their results to select an H59 arm.
- Only an arm that passes its admission rule may receive one terminal `DL-Val` read per seed, after the fold-B decision is frozen. No repeated DL-Val reads, checkpoint selection on DL-Val, sealed/meta-test families, outer folds, holdout labels, or live-loop state are permitted. If no arm passes, do not read DL-Val.

## Fixed training protocol

Run three paired seeds: `12`, `13`, and `14`. Each arm/seed starts fresh from the same r882 adapter values; no weights or optimizer state transfer between seeds or arms. Train only the adapter parameters; freeze DOPE and every non-adapter parameter. Before and after every cell, verify the DOPE hash above and the frozen/trainable parameter inventory. A mismatch ends the cell without repair or resume.

Use a fixed horizon of 20 rounds and 2 optimizer steps per round (40 updates per arm/seed), batch size 256 contexts, slate size 192, AdamW learning rate `3e-4`, weight decay `1e-4`, cosine decay to `3e-5`, gradient norm clip 1.0, and no early stopping. The same context and slate sequence is replayed across all four arms for each seed. The total outcome budget is 12 cells and 480 adapter optimizer updates. Round/step caps govern completion; elapsed time cannot extend or shorten a cell. Report all three seeds, including failed or incomplete cells, without replacement.

## Metrics, admission, and interpretation

For each arm, report the paired candidate-minus-r882 family utility gains on fold B, per seed and pooled across seeds. Report family-grouped Bradley–Terry Elo versus r882, W/T/L, family median gain, and a family-cluster bootstrap 95% confidence interval for the pooled family median. For admission, use 10,000 paired family-cluster bootstrap resamples, sampling family IDs and retaining each family's observations across all three seeds together. Report the one-sided mean-gain LCB95, family median, and fraction of families whose seed-median gain is less than `-0.01`.

An arm is admitted only if all three conditions hold: (1) mean-gain LCB95 is strictly greater than `0.001`; (2) family median gain is nonnegative; and (3) no more than 10% of families have seed-median gain below `-0.01`. Ties fail. Apply the rule independently to each arm; do not choose a winner by comparing fold-B scores. A result may be called a **breakthrough** only if the pooled family-median gain has a two-sided 95% family-cluster bootstrap interval wholly above zero and each of the three seed-specific family-median gains is also positive with its own 95% family-cluster bootstrap interval wholly above zero. Otherwise describe it only as admitted (if it passes the gate) or not admitted.

## Resources, ordering, and stop rules

This document and its review are CPU-only; do not run training, GPU work, or read dataset payloads. Before future outcome execution, wait until both H55 and H57 are closed, then queue H59 after them. Each GPU cell must hold `~/jope_plan/qualification.lock` exclusively, record lock wait, device and free memory, and release the lock between cells. Verify CPU utilization is at most 85% before launch and preserve the fixed round/step budget. Do not interrupt another lock holder, fall back to CPU training, or start a cell when the resource guard is not met.

The planning estimate is **about 60 GPU-minutes for outcome cells, plus setup and final scoring**: 480 capped optimizer updates at a provisional 7.5 seconds/update, plus 15 minutes aggregate for paired fold-B/eligible terminal validation scoring. This is an explicit planning assumption, not a measured runtime; no H55/H57 timing receipt establishing H59 throughput was found during preregistration. Before training, perform a timing-only measurement that produces no outcome, then record the observed projection and confirm the 60-minute cell budget can be met without changing the fixed update count. If it cannot, stop and seek an owner amendment before any outcome cell.

## Required receipts

Record source commit, checkpoint/source/fold hashes, arm and seed, exact completed rounds/steps, sampling identity, optimizer settings, trainable parameter inventory, DOPE hashes before/after, resource readings, per-family paired gains, bootstrap seed and resamples, Elo implementation/version, all exclusions, and whether DL-Val was eligible and read. Record zero sealed/holdout payloads opened and zero outer rows delivered. Keep compact receipts and reports outside large-data paths; do not add data or checkpoints to Git.
