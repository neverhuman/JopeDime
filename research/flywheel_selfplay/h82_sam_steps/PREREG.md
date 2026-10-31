# H82: prospective SAM/ASAM/K8 continuation

Registered before any H82 timing, training or outcome cell, under owner
`sol_v1_hwv_h82_sam_continuation_t0030.md` (2026-10-31). H76 remains incomplete;
no H76 outcome cell or score is read or reused. This is a new conditional r882
dev experiment, not an unseen-family generalization or serving claim.

Science follows [H76's frozen protocol](../h76_sam_steps/PREREG.md): real-S3
H50 packet (all 1,219 DL-Train contexts/205 families; terminal 91 dev/19 disjoint
families), same seed-4900 fold lineage, checkpoint COPY SHA
`0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64`,
frozen DOPE `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`,
exactly 29,211 non-anchor adapter parameters. Preserve the manifest, selected
roster and fused-source pins from the unchanged H76 loader. Train roster includes
the historical fold B; previously reused dev remains exploratory.

Fresh paired seeds **24,25,26**; arms in order CONTROL (2 AdamW updates/round),
K8 (8), SAM (2, rho .05), ASAM (2, rho .5, weight-only eta .01). Each starts
fresh from r882 and runs20 rounds, batch16, slate192, constant lr1e-4, wd1e-4,
final-gradient clip1. Reuse H76's actual batch planner/optimizer updates and H59
CONTROL SmoothL1+0.1KL loss unchanged. Matched first-two-step context/candidate
streams, replayed SAM dropout RNG, exact perturbation rollback, one AdamW/decay
application. No scheduler, early stopping, dev selection or arm/seed replacement.

Only the infrastructure retry policy changes: a host-LOAD guard abort **before
any full train/dev scoring begins** may retry at most **two additional times**
(three attempts total per logical cell). A durable marker flips before the first
scoring operation; missing/uncertain markers, integrity/RSS/memory/disk failures,
and aborts after scoring starts cannot retry. Minibatch optimization loss is not
a terminal outcome. Each retry restarts r882/optimizer with the same seed and
fixed batch stream; discard the void's weights. No retry after any outcome,
no best-of selection. Every attempt/void/setup/evaluation is charged against one
non-resettable **2,700s GPU-cell-wall cap**, with10s shutdown reserve. Unknown
in-progress launches forbid resume. No replacement of a completed outcome.

Before every attempt's FIFO ticket, require load1 <=85/102 on two consecutive
checks30s apart, backing off at most30min outside the GPU lock. FIFO helper
`qualification_lock_fifo.py` via merged H61 acquisition, wave_name H82; W1/H80
older tickets go first. Fresh ticket per attempt/cell; release promptly before
aggregation, >=30s yield and legacy backoff. nice19, OMP/MKL4, BLAS1, RSS<6GiB,
MemAvailable>=16GiB, launch disk<94.5%, writer<95%, >=1GiB free. In-cell guards
unchanged. Lock/CPU waiting is separate, cumulatively capped at3h. Never touch
other jobs, live xbabe1, xbabe0/3, sealed/holdout/outer payloads.

One initial K8/seed24 timing cell runs20 rounds plus train-only evaluation and
discards its scores/weights; no dev read. Before outcomes require prior void wall
+ successful timing wall*13*1.25 <=2700s. Then12 fixed cells, seeds24,25,26 each
CONTROL,K8,SAM,ASAM. Per-cell DOPE/inventory checks before/after, fixed-final
unrevealed-only selection4..P-1, clip passing utility[-2,4], failed gate=-2.
Each CONTROL cell additionally scores its original r882 adapter on the same dev
packet after candidate scoring, then restores candidate weights. These three
fresh reference scores supply same-seed r882 Elo; they never select a candidate.

Primary remains per-seed equal-family val_reg gain versus fresh CONTROL, median
of3. Bootstrap10,000 draws of19 family clusters, retaining complete3-seed vectors,
fixed RNG76000 and common draws. Report two-sided95% percentile CI and one-sided
LCB95 (5th percentile); admit only LCB95 >.001546 strictly AND median>=0; ties
fail. Require all12 complete paired cells; missing/void is unknown, never zero.
Three nominal contrasts; retain Bonferroni LCB diagnostic without changing gate.

Report train/val gaps; family-grouped Elo versus CONTROL1500 and versus r8821500
per seed and median across seeds, with paired-family bootstrap95% intervals.
Within-seed family utility wins count1, ties.5; Elo delta=400log10(p/(1-p)),
endpoint infinities serialized as strings. CI for median Elo transforms the
bootstrap median win rate (monotonic equivalence for3 seeds), avoiding infinity
interpolation. Also retain H76-style pooled family/seed Elo explicitly labeled.
No AUC. CPU PR/checks/merge precede FIFO launch; closeout README/metrics/one
index row goes to super_merger_codex. No large objects or CI dispatch.
