# H93 — Lookahead, annealed gradient noise, and effective batch 64

**Status:** preregistered before H93 timing or outcomes. Owner `luna_v1_hwv03`; assignment is `~/nudges/h93_lookahead_noise_t0205.md`. H93 is a conditional candidate-generation experiment from an isolated copy of checkpoint r882. It does not qualify serving or authorize changes to the live flywheel.

## Question and lineage

H61 accepted 0/240 candidates, making candidate generation the bottleneck. H46 already tried warmup/cosine schedules and family-balanced loss on a restore-on-reject plateau; H49 and H51 flagged near-pseudoreplicate seeds. H76/H82 established the fixed 20-round, two-AdamW-updates-per-round control; H90 separately studies family-balanced sampling, target smoothing, and learning-rate restart recipes. H93 holds that control fixed and tests Lookahead synchronization, annealed gradient noise, and effective batch size. Its fresh paired seeds and identical-seed utility flag make seed dependence visible; the top-four overlap and adapter-distance readouts describe diversity without changing the admission rule.

The data lineage is unchanged from H82: the real-S3 H50 packet (1,219 DL-Train contexts, 205 families), its 91 dev contexts across 19 disjoint families, and the seed-4900 fold lineage. Reuse the H50 manifest, selected roster, fused source, and scorer pins. Each cell loads a verified COPY of r882 (`0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64`); only the 29,211 non-anchor adapter parameters train. Check the frozen DOPE hash `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814` before and after each cell. H93 does not read or reuse H76/H82 outcome arrays.

## Fixed protocol and arms

All arms use 20 rounds, two optimizer updates per round (40 updates total), slate 192, AdamW weight decay `1e-4`, final gradient clip 1, and the unchanged CONTROL loss (SmoothL1 + `0.1 * KL`). Every outcome cell starts from a fresh r882 copy. No early stopping, dev selection, arm replacement, best-of selection, or additional optimizer update is allowed.

- **CONTROL:** H82 CONTROL exactly: batch 16 and AdamW learning rate `1e-4`.
- **LA:** the same two fast AdamW updates per round at `1e-4`; Lookahead `k=5`, `alpha=0.5`. Fast optimizer moments and slow weights persist through all rounds; synchronize slow weights after fast updates 5, 10, …, 40.
- **GN:** the same CONTROL updates. Add independent seed-keyed Gaussian noise to adapter gradients before clipping, with variance `sigma_t^2 = 0.01 / (1+t)^0.55`, where `t=0..39` is the update index. A cell's noise stream is fixed by its seed.
- **EB4:** four consecutive microbatches of 16 are gradient-accumulated into each update (effective batch 64); keep 40 updates and set AdamW learning rate to `2e-4` (`sqrt(4)` scaling). Average the four microbatch losses before clipping and the single optimizer step.

Fresh paired seeds are **30, 31, 32** across all four arms. The sole timing cell is EB4/seed30: 40 train updates plus train-only timing, with score summaries and weights discarded. It is not an outcome cell. Twelve outcome cells are fixed as CONTROL, LA, GN, EB4 for each seed.

## Readouts and admission

The primary is the per-seed equal-family dev `val_reg` gain versus same-seed CONTROL, summarized by the median of three seeds. Use 10,000 common family-cluster bootstrap draws over the 19 dev families with fixed RNG seed `93000`; report the two-sided 95% percentile interval and one-sided LCB95. Admit a candidate arm only when `LCB95 > 0.001546` strictly and median gain is nonnegative. Ties fail. Missing or void cells are unknown, never zero; require all twelve paired outcome cells.

Report family-grouped Elo versus same-seed CONTROL (rated 1500) and the fresh r882 reference, per seed and median with paired-family 95% intervals. Report train/dev means and train/dev gaps. Do not report AUC. Match H90's pseudoreplication check: report exact repeated context-utility, family-utility, and scalar equal-family-utility seed groups; flag any repeated utility as a warning, never as independent seed evidence.

For each arm, report (1) mean per-context overlap fraction of the four lowest predicted-cost candidates among unrevealed indices 4..P-1 against fresh r882 on the same dev contexts (intersection size divided by four), per seed and median; and (2) the L2 distance from the initial r882 adapter, per seed and median. These are descriptive diagnostics only.

## Resource, integrity, and launch gates

Reuse H82's FIFO helper and pre-scoring load retry policy with `wave_name="H93"`; its live ticket must remain behind older H80/H82/W1/H90 tickets. The H93 source additionally refuses to launch unless it finds a complete `jope.h80.closeout/v1` receipt at `~/jope_plan/ns/receipts/H80_closeout.json` **and** a live H90 ticket in `~/jope_plan/qualification.queue/`. Do not create the H93 ticket until both exist. CPU preregistration, synthetic tests, and `--cpu-dry-run` do not touch either gate.

Keep H82 limits: cumulative GPU cell-wall cap 2,700 seconds including timing, voids, setup, and retries; cumulative resource/lock wait cap three hours; no more than two extra pre-scoring load retries per cell; a durable scoring marker forbids retry after any scoring begins. Require load <=85/102 on two checks 30 seconds apart, nice 19 GPU children, OMP/MKL 4, BLAS 1, RSS <6 GiB, MemAvailable >=16 GiB, launch disk <94.5%, writer disk <95%, and >=1 GiB free. A failed disk/load/memory guard waits or stops according to the fixed H82 policy; delete nothing and do not control other jobs.

Never access sealed, holdout, meta-test, outer, or live-loop payloads; never touch xbabe0, xbabe1's live run directory, or xbabe3. Do not acquire a GPU lease, enqueue a ticket, or run a timing/outcome cell before the H80 closeout and H90-ticket gates pass. Do not change Cargo.lock or DOPE. No large artifacts or outcomes belong in the source PR.
