# H105 — shrink adapter drift toward r882

Owner: sol_v1_dsc08. Authority: `h105_shrink_t0505.md` and H105 namespace receipt.
CPU preparation only: source, mathematical fixtures and a data-free dry-run.
No H105 timing, optimizer run, data load, FIFO ticket or GPU waiter is authorized
by this task. Hand the CPU PR to super_merger_codex, then stop.

## Hypothesis and prior-only calibration

H97's 40-update fine-tunes remained below r882. Its EMA candidate drifted less
than CONTROL and was less harmful. This motivates linear interpolation toward
r882 and a fixed L2-to-r882 anchor; it does not establish drift as the cause.
[H90](../h90_candidate_diversity/README.md),
[H49](../../../docs/research/studies/dope-neural-ranking-20261021/h49/README.md)
and [H51](../../../docs/research/studies/dope-neural-ranking-20261021/h51/README.md)
warn against treating different adapter hashes or repeated utilities as independent
confirmation. No new H105 outcomes inform these choices.

The only calibration input is H97 aggregate
`/home/ubuntu/jope-data/h97/h97-wave/metrics.json`, SHA256
`109e030f7f8f57b604ad5f2d8df576a817fca0ae482c08026398eca3338d3054`.
Its `arms.EMA.adapter_l2_from_r882_median` is `0.34640736440899245`.
Fix anchor lambda = clip_norm / prior_EMA_drift =
**2.8867746553428666**. The extra loss is
`(lambda / 2) * sum((theta - theta_r882)**2)` over non-anchor adapter
parameters, without parameter-count normalization. Its gradient has norm one
at that prior drift, matching the unchanged final clip bound. This is one CPU
arithmetic choice from prior drift, with no sweep, fitting or new utility read.

## Inputs and fresh family split

Use only the real-S3 H50 DL-Train packet: 1,219 regression contexts /205 families.
The pinned roster's `train` member supplies the sole payload closure; shared
roster identity metadata is checked, but DL-Val rows/payloads/scores never enter
the worker. The 19 H97 DL-Val families are disjoint from these 205 and are never
used for H105 admission. No sealed, holdout, outer, meta-test or live-loop reads.
CPU preparation/dry-run opens neither the roster nor any data/checkpoint.

Fresh whole-family split: sort family IDs, permute with NumPy default_rng
**105042**, split into five equal folds. Fold0 (41 families) is assessment;
folds1–4 (164 families) train. Context counts remain UNKNOWN until a future
approved load. This differs from seeds4900/9900. All arms/seeds share this split.
Only training families supply gradients, targets and candidate slates. Timing
scores training families only. Assessment is scored once at the fixed terminal
candidate, with a fresh r882 reference on the same families. Never choose a fold,
checkpoint, alpha, lambda or seed from outcomes.

r882 previously trained on all1,219 contexts, including these assessment families;
this is conditional development with baseline exposure, not unseen-family
qualification. A new family assignment does not undo that exposure.

Checkpoint COPY SHA256:
`0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64`.
Frozen DOPE:
`c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`.
Check before/after each cell; exactly29,211 non-anchor adapter parameters train.
Use only fresh copies and unique `scratch/<arm>-seed<seed>-attemptN/` paths;
refuse existing attempt directories, never overwrite/chmod a staged input.

## Fixed arms and compute

Paired fresh seeds **42,43,44**. Reuse H97 CONTROL/H76/H59 numerical recipe:
20 rounds x2 AdamW updates,40 total, batch16, slate192, lr1e-4, wd1e-4,
clip1, SmoothL1(beta1)+0.1KL(temperature0.2). No dropout, EMA or LR search.
The mandated train-family split is the only shared data change from H97.

- CONTROL: train40 updates and score the final fine-tuned adapter (alpha1).
- INTERP025: identical unanchored training, then `r882 + .25*(fine_tuned-r882)`.
- INTERP050: identical unanchored training, then `r882 + .5*(fine_tuned-r882)`.
- ANCHOR: add the fixed penalty above during the same40 updates; score its final
  adapter without interpolation.

Interpolation changes no optimizer update or DOPE/anchor tensor. Endpoints preserve
exact reference/fine-tuned bits. The three unanchored arms must have identical
pre-interpolation fine-tuned digests within each seed; otherwise the matched panel
is incomplete, not repaired by another run. All12 unique arm/seed cells required.
Order: seed-major, CONTROL/INTERP025/INTERP050/ANCHOR. One discarded ANCHOR/42 timing
cell precedes them,40 updates plus train-only evaluation, no assessment read.

## Fixed readout and admission

Reference is **r882**, never CONTROL. Per seed compute the equal-family mean
paired assessment gain, then take the median of three seeds. Use10,000 common
family-cluster bootstrap draws over all41 assessment families, RNG **105000**,
retaining full paired seed vectors. Report two-sided95% percentile CI, raw
one-sided LCB95, per-seed/median Elo versus r8821500 with family-cluster95% CI,
train/assessment gaps, adapter L2 and exact utility seed-pseudoreplication groups.
Identical seed utilities are flagged, never independent confirmations. No AUC.

There are **four** planned r882 comparisons, including CONTROL. Order arms by
bootstrap lower-tail area `(1 + count(boot_stat <= .001))/(10000+1)`; break ties
in declared arm order. Holm step k uses tail probability `.05/(4-k)` for k=0..3.
Its one-sided lower bound must strictly exceed **0.001**, and all preceding steps
must pass; report adjusted tail areas and stepwise lower bounds. This is the
preregistered percentile-bootstrap/Holm approximation, not an exact finite-sample
test. Treat differences <=1e-12 from the margin as ties; ties fail.

Also require a nonnegative primary median and harm <=10%: a family is harmed if
its mean gain over seeds is <-0.01 versus r882 (the existing H100/H57 convention).
Report per-seed harm and any-negative-family fractions descriptively. Missing,
duplicate, nonfinite or identity-mismatched cells/families are INCOMPLETE/UNKNOWN,
never zero or ADVANCE. No admission from a subset or synthetic fixtures.

## Future launch constraints; no launch in this task

CPU PR must merge before a later owner-authorized launch from merged source.
FIFO order is H99 then H100 then H105. Validate older live FIFO tickets or actual
prior GPU-start metadata before creating an H105 ticket; do not invent an extra
full-closeout approval requirement. This CPU dry-run creates no ticket or waiter.

One non-resettable2700s GPU-cell-wall cap includes setup, scoring, discarded timing,
all voids/retries;10s shutdown reserve. Timing gate: prior voids + timing_wall*13*1.25
<=2700. At most two additional retries, solely host-load aborts before durable full
scoring starts; all other failures stop, unknown in-progress launches forbid resume.
Preserve cumulative3h resource/lock wait, nice19, OMP/MKL4, BLAS1, host load/CPU<=85,
RSS<6GiB, MemAvailable>=16GiB, launch disk<94.5%, writer<95%/>=1GiB free. Failed
launch guards WAIT; delete nothing, never control another job or change the live loop.
