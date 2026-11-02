# H230 — COSWR × GATED_025 stack

Owner `hw_t1945_astra_h224close_h230.md`; no H230 model fit or outcome has run.
Merge this preregistration by2026-11-02T22:00Z before any real TRAIN smoke or
primary attempt. H224 closeout merged as#2042 before H230 work started.

## Hypothesis and four arms

H136 COSWR reduced adapter drift and beat CONTROL on2/3 seeds, but did not beat
unchanged r882. GATED_025 repeatedly improved native means without clearing the
DL-Val plateau margin. H230 tests whether their effects combine; prior partial
signals are not evidence of an independent validation win.

- R882: unchanged native reference.
- COSWR: H136 frozen40-update adapter recipe, trained on the current FIT families.
- G025: H217 frozen BLEND_025/zgap/tau rule with r882 costs as reference.
- STACK: the same rule with the COSWR adapter costs as reference; tau chosen
  anew on FIT only, comparing blend utility with COSWR utility inside FIT.

COSWR and STACK share each trained adapter; G025 and STACK share every HistGBM.
No lambda, schedule, update count or other hyperparameter is selected on A/DL-Val.

## Cohort, folds and source identities

Use the SHA-pinned H124 real-S3 TRAIN packet:205 families/1,219 contexts, with
separate91 DL-Val contexts/19 families. [provenance.json](provenance.json) binds
the catalog and historical dataset IDs. No sealed, holdout, meta-test or original
outer-test payloads; only the fixed four revealed labels may enter a prediction.

Seeds230101–103, fold seeds230141–143. Sorted families are permuted by each
fixed NumPy default_rng fold seed and array_split into five41-family groups.
Cell k has A=k, B=(k+1)%5, FIT=the other123 families; B is unused. Every context
is assessed once per seed. [fold_manifest.json](fold_manifest.json) reports
Jaccard against historical partitions; no seed is rejected/reselected using
performance or overlap. These are reused development families, not a fresh
unseen-family qualification cohort.

H224 execution commit e3c978d5d36f50a190b82543e592834b549e0844 supplies the
frozen H217 reference gate/statistics and corrected strict plateau arithmetic.
The H217/H124/H139 data and HistGBM helpers remain SHA-pinned to the historical
parent in [manifest.json](manifest.json), independently of later main refactors.
H136's executed recipe has SHA
b0eda24a960f31c52fc308f5b667f1b76c9a6a55dd0ce031898beeda4fc21072.
Only its named schedule, batch-plan, loss and frozen-parameter functions are
loaded; its old launch/resume/assessment controller is never executed.

## Training and gate protocol

Each of15 seed/outer cells starts from a fresh in-memory copy of SHA-verified
r882, never the live loop/run directory. Only47 non-anchor adapter tensors,
29,211 parameters, train. All other parameters are frozen and hash-checked.
DOPE c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814 is
verified before/after; CUDA must remain uninitialised.

Preserve H136 COSWR exactly:40 AdamW updates, lr peak1e-4, weight_decay1e-4,
gradient norm clip1, one16-context microbatch/update,192 candidates/context;
the inherited CONTROL sampler rotates uniform FIT context IDs and chooses the
best48 plus144 sampled candidates. Its original RNG stream consumes all eight
batch draws per round while using two; this is preserved, not simplified.
Four-update linear warmup starts at2.5e-5, then cosine warm restarts T0=40/3.
The frozen target clipping/failed-gate cost and smooth-L1+.1KL objective remain
unchanged. The H48 predictor's first four sampled training-slate labels stay
unchanged. Program/projection arrays load lazily to fit the4GiB cap; fixture
checks compare dense/lazy tensors and revealed-label order.

The adaptation from H136 is the registered H217-style cross-fit partition:
123 FIT families per cell instead of H136's original164; the optimizer recipe
and40-update budget are unchanged. There are15 adapters/600 total updates.
Each optimizer update is fsynced to a journal; an interrupted in-flight step
without a durable record remains unknown, never silently charged as zero.

HistGBM uses the frozen H217/H124 recipe (max_iter300, inherited early stopping,
learning_rate.08,63 leaves,L2=1). For each seed, fit five outer models excluding
A/B and ten distinct inner models excluding A/B/one FIT group:45 unique fits,
60 logical cell uses. Candidate sample seed230140 selects192 unrevealed rows
per context. Inner FIT-family OOF predictions supply calibration and tau input;
no HistGBM predicts its own training family for this selection.

G025 and STACK use identical FIT-only calibration, lambda.25, zgap, tau grid,
<=5% FIT-family harm constraint and conservative tie/closed fallback as H217.
COSWR's reference costs inside FIT are in-sample for the adapter; they are not
claimed to be adapter-OOF. This can make the selected STACK threshold optimistic;
A families remain excluded from both adapter training and threshold selection.
The pooled-A primary evaluates that full procedure. The gate label is not an
extra learned model in H230.

## Native regression, smoke and global lock

First recompute unchanged r882 on91 DL-Val contexts: every native utility and
mean1.1377941898164525 must match bit-exactly, before any fit/update. Then run
one non-scored TRAIN smoke consisting of the first four registered HistGBMs
and first40-update COSWR cell, including FIT-only calibration/tau selection.
Reuse all five artifacts in the fixed primary inventory; no additional fit or
optimizer update is granted by smoke success. Smoke failure ends the sole run.

Complete45 HistGBMs and15 adapters, their cost predictions, calibration and
15 pairs of taus. Fsync/archive a global stage1 lock before any pooled-A or
treatment DL-Val utility scoring. Predicting treatment costs beforehand is
permitted; those computations receive only features and four revealed labels.
All trained parameters and prediction artifacts are hash-bound in cell receipts.
No step/checkpoint/tau is selected using a treatment outcome.

## Endpoints and strict decision

Primary STACK minus R882: median of three equal-family pooled-A seed means,
10,000 paired family-cluster percentile95% CI. It passes iff median>.002,
LCB>0, harm<=10% (a family's three-seed mean loss worse than.01), and at least
2/3 seed gains>0. A1e-12 numerical exclusion band rejects ties and near-ties
at the gain/LCB/positive-seed boundaries; harm's inclusive10% cap is unchanged.

Co-primary: average the15 seed/cell panels for each of91 DL-Val contexts,
then compute context-mean native gain and10,000 family-cluster bootstrap95% CI.
It passes iff gain>.002 and LCB>0, with the same1e-12 exclusion band. ADVANCE
requires both endpoints; otherwise NO_ADVANCE. No serving promotion follows
from this development-cohort result.

Report all four arms' medians/CIs, per-seed signs, harm, adoption and family-paired
Elo with r882=1500 (1,000 draws). Report STACK−G025 as requested and the full
2×2 interaction (STACK−COSWR)−(G025−R882), both descriptive, with paired CIs.
Bootstrap seeds230150(native),230151(Elo),230152(DL-Val). Per-seed DL-Val
estimates and their median are descriptive; they do not replace the registered
context-mean co-primary. No AUC, speed or favorable-seed verdict.

## Resources, custody and stop behavior

xbabe2 CPU only; fixed cores[2,22,67,104], nice19, complete-tree RSS<4GiB,
MemAvailable>=20GiB, free disk>=300GB, own output<2GB. Load5>60 pauses only
the own worker; recheck every60s and resume at<=55. Memory/disk/deadline guards
remain hard stops while paused. No GPU or qualification lock.

One create-only primary attempt, no retry or continuation. Record the host boot
ID in the claim. A reboot interruption is INTERRUPTED and must be closed from
durable receipts, never restarted. A missing terminal after a changed boot ID
is not success. Compute stops absolutely at2026-11-03T03:30Z; closeout by04:00Z.
Failure to meet a deadline closes INCOMPLETE. Pauses never extend the clock.

Before launch preserve merged source, H136 recipe, checks and input identities
in create-only S3 keys. Every fit/adapter/cell/lock/panel/metric upload is
recovered by exact VersionId/SHA. Full outputs stay outside git. Closeout
publishes README, small metrics, both index rows, and custody/resource receipts
through a small reviewed PR with targeted local tests and squash merge.
