# H191 — nested kernel coefficient selection, zero updates

Owner: rd_astra / [ASTRA]. Authority: `/home/ubuntu/nudges/hwave_t0835_h191.md`.
H186 closeout #1567 (`d4e8602318754e7d50c97c1a63afe2949bf400bd`) motivates a new
prespecified nested policy; H186 itself is complete and will not be rerun.

## Intervention and input closure

Keep the r882 learned adapter and DOPE weights byte-identical. Vary only the multiplier
of the existing `0.45` kernel coefficient over **1.0, 1.5, 2.0, 2.5, 3.0, 4.0**.
Multiplier 1 is native R882. Null weight, top-k, attention, priors, scale, history,
full candidate pools, first-four hash-ordered reveals, and first-argmin ties stay fixed.
No optimizer construction, backward, fitted models, or replacement arms.

Use the pinned real-S3 corpus lineage already admitted by H186: 1,219 TRAIN contexts /
205 families and exactly 91 development contexts / 19 families for the explicit native
baseline and descriptive terminal readout. No sealed/holdout/meta-test or outer-test
payloads, no live-loop writes, no xbabe0/xbabe3, no GPU. Inputs, inherited assets,
source files and evaluation identity are SHA-pinned in `manifest.json` and
`provenance.json`; cached features include the disclosed development-feature rows.
Historical exposure of r882 assets and these cohorts is not erased by reshuffling.

## Nested policy and stage order

Seeds **191101, 191102, 191103** use literal, distinct fold seeds **191141, 191142,
191143**, respectively, through imported H177 family partitioning. Each has five cyclic
123-FIT / 41-B / 41-A family cells, with exact-once B and A coverage. FIT is not used
to update or select anything in this zero-update wave.

1. The native 91-context r882 baseline must pass **bit-exactly**, including its full
   per-context vector and mean **1.1377941898164525**, before TRAIN scoring.
2. For each `(seed, outer fold)`, score the fixed grid on that cell's B only. Average
   original utility within each family, then equally across B families. Choose its
   argmax; exact ties choose the smaller multiplier. No A outcome enters this choice.
3. Complete all 15 B cells and fsync `kx_selection_lock.json` with every choice, B mean,
   B-panel SHA, fold-manifest SHA and adapter SHA before any A or treatment DL-Val
   scoring. The scorer refuses missing/incomplete/mutated locks and B reads after lock.
4. Score each locked choice once on its corresponding A. Pool the resulting **KXSEL**
   vectors, paired against native R882, across all five folds for each seed.
   Fixed KX2.5/KX3.0/KX4.0 pooled A readouts are **descriptive dose responses only**.
5. For descriptive DL-Val, each seed uses the **modal multiplier among its five locked
   B choices**, with ties toward the smaller multiplier. This policy is also fixed in
   the B lock. There is no DL-Val tuning or ensemble intervention. Score the same three
   fixed descriptive multipliers alongside it; no descriptive result can rescue failure.

H186's `Engine.evaluate_many`, component scorer and output arithmetic are imported
verbatim through private module bindings. Only its grid/constant/guard configuration is
rebound. The H177 selector, decision-rule object, partition/coverage checks, verdict
resolver and committed H178 receipt are imported; no decision function is reimplemented.
H177's historical function names and literal `m=4` metadata are retained in provenance;
the actual one-arm call computes Holm with **m=1**. Candidate version 1 is not an update.

## Primary decision and uncertainty

The **one primary arm is KXSEL**. Pass its pooled-A panel through the unchanged H177
selector: seed-averaged family gain strictly > +0.001 (implementation tolerance 1e-12),
paired family CI lower > 0, and Holm-m1 p < .05. Additionally require the **median
seed gain > +0.001** (same strict tolerance) and its **median-seed CI lower > 0**.
ADVANCE requires those conditions and the imported H178 receipt gate. Otherwise retain
the imported NO_ADVANCE/DIAGNOSTIC disposition. Ties reject; no adoption is authorized.
On ADVANCE, flag the flywheel owner for separate validation and DOPE checks.

Bootstrap seed **191150**, 10,000 paired family draws shared across seeds; average
contexts within family, resample families, and reduce the three seed statistics with
`median(axis=0)`. Report all per-seed gains, median/95% CI, the actual imported primary
decision, and exact -2.0 pick counts separately from gate-invalid counts. Report canonical
400-iteration Bradley–Terry Elo with prior .01 and utility tie tolerance 1e-4, R882=1500
in the five-policy readout, with 1,000 paired-family draws (seed191151) and median-seed
Elo intervals. Higher utility wins; mean utility magnitude and win frequency differ.
Report context-weighted utility/val_reg separately from equal-family effects. No AUC
or speed outcome. Report all 15 KX choices, counts and descriptive mode stability.

The fixed-grid context utilities are identical across seeds; only the B/A partition and
thus selected policy can vary. **Three positive seeds are not independent replication**
and are not an extra gate. A_k is excluded from its own B_k choice, but families recur
in other folds and were exposed by H186. The family CI conditions on locked fold
policies; it does not reselect B inside bootstrap draws. This is not pristine holdout
confirmation or a full nested-procedure uncertainty guarantee. H178 calibrated the
original four-arm rule, not this nested one-arm procedure; importing its receipt does
not establish a new empirical FWER guarantee for H191.

## Execution and preservation

One scientific attempt, **3,600-second hard wall cap**, CPU xbabe2 nice19, math threads1,
aggregate <=4 threads/RSS<6GB. Stop if load >85% of online cores, host CPU>85%,
MemAvailable<16GiB, user-visible disk usage>=93.8%, or free<40GB. The complete owned
scientific output root (including inputs/logs) must stay **<500,000,000 bytes**.
No threshold relaxation, adaptive extension, grid changes or scientific retry.
Constructor smoke scores zero contexts and does not claim the scientific attempt.

Retain raw per-candidate grid costs/choices for every B/A/Val cell, selected panels,
stage lock, baseline, source/input identities, audit and resource receipts outside Git.
Metrics remain <100KB; checkpoints and pools never enter Git. Reconstruct selections,
original labels/gates, primary panels and registered statistics after the single run
without model execution. Preserve failures as INCOMPLETE, never a false efficacy null.
Create-only S3 custody uses exact-version payload/member recovery before owned bulk
reclamation. Preregister, request rd_argon review (bounded15minutes), pass targeted
tests/Ruff/records, merge, execute once, then merge the closeout and remove the lane.
