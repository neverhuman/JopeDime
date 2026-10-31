# H97 — Adapter regularization and final EMA candidate

**Status:** preregistered before any H97 timing or outcome. Owner `luna_v1_bat02`; renamed from H95 because #963 retains H95 for W8 and #964 retains H96 for W4. H97 is a candidate-generation study only. It does not qualify serving or authorize changes to the live flywheel.

## Question and lineage

H61 accepted 0/240 candidates, pointing to candidate generation as the bottleneck. H46 already covered long warmup/cosine and EMA recipes; H59 covered clipping and normalization; H76/H82 covered SAM/ASAM/K8; H80 covered early stopping; H90 covered family-balanced sampling, target smoothing and learning-rate restarts; H93 covered Lookahead, gradient noise and effective batch. H97 isolates the remaining r882 candidates: stronger decoupled weight decay, additional adapter hidden-activation dropout, and a Polyak/EMA final candidate under the unchanged 40-update budget.

The data lineage is exactly H82/H93: the real-S3 H50 packet (1,219 DL-Train contexts across 205 families), its 91 dev contexts across 19 disjoint families, and seed-4900 family-fold lineage. Reuse the pinned H50 manifest, selected roster, fused source, H82/H93 loaders and scorer. Each cell starts from its own byte-verified copy of r882 (`0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64`). Only the 29,211 non-anchor adapter parameters train. Verify frozen DOPE SHA `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814` before and after every cell. No H76/H82/H93 outcome arrays are inputs to H97.

Never access sealed, holdout, meta-test, outer, or live-loop payloads. Do not use xbabe0, xbabe3, or xbabe1's live run directory. No CUDA run or data payload was used for CPU preregistration.

## Fixed protocol and arms

Every scored cell uses exactly 20 rounds × 2 AdamW updates per round (40 updates), batch 16, slate 192, final gradient clip 1, and the unchanged H82/H93 loss (SmoothL1 + `0.1 * KL`). No early stopping, dev selection, arm replacement, best-of selection, or extra optimizer update is allowed. Every cell starts from a fresh r882 copy.

- **CONTROL:** H82/H93 CONTROL exactly: AdamW learning rate `1e-4`, decoupled weight decay `1e-4`.
- **WD:** CONTROL with decoupled AdamW weight decay `1e-2`; all other settings unchanged.
- **DROP:** CONTROL with `p=0.1` dropout applied to the adapter's hidden activation outputs only while training. The frozen anchor stays in eval mode. The adapter contains SiLU hidden activations; install dropout hooks on those hidden activations, and disable them before all scoring so evaluation is deterministic. Do not change the existing modality-dropout setting or any parameter count.
- **EMA:** CONTROL updates with final candidate weights set to an exponential moving average of adapter parameters. Initialize the shadow from the fresh r882 adapter; after each of 40 optimizer steps apply `ema = 0.9 * ema + 0.1 * current`. Score/save the EMA adapter as the sole candidate; do not select between raw and EMA weights.

Fresh paired seeds are **33, 34, 35** across all four arms. The sole timing cell is DROP/seed33: 40 training updates plus train-only timing, with weights and score summaries discarded. It is not an outcome cell. Twelve outcome cells are fixed as CONTROL, WD, DROP, EMA for each seed.

## Readouts and admission

The primary is each candidate's per-seed equal-family dev `val_reg` gain versus same-seed CONTROL, summarized by the median of three seeds. Use 10,000 common family-cluster bootstrap draws over the 19 dev families with fixed RNG seed `95000`; report two-sided 95% percentile intervals and one-sided LCB95. Admit an arm only if `LCB95 > 0.001546` strictly and median gain is nonnegative. Treat differences within `1e-12` of the threshold as ties to avoid floating-point roundoff making an exact tie pass. Ties fail. Require all twelve paired outcome cells; missing or void cells are unknown, never zero.

Report family-grouped Elo versus same-seed CONTROL (rated 1500) and versus fresh r882, per seed and median with paired-family 95% intervals. Do not report AUC. Report exact repeated context-utility, family-utility and scalar equal-family-utility seed groups; repeated utilities warn of pseudoreplication and never count as independent seed evidence. Report adapter L2 distance from initial r882 and top-four overlap with fresh r882 on the same dev contexts, per seed and median, as descriptive diagnostics only.

## Staging, integrity and launch gates

Stage the immutable r882 copy under a unique attempt directory `scratch/<cell>-attemptN/`; never reuse a previous attempt directory or overwrite an existing staged copy. Add a test proving different attempts resolve to different paths and a repeated attempt path is rejected.

No H97 qualification ticket is created before H93 has a live FIFO ticket. Preserve FIFO precedence H80 → H90 → H93 → H82 → H97. The cumulative GPU cell-wall cap is 2,700 seconds including timing, setup, voids and retries; cumulative lock/resource wait cap is three hours. Before launch require host load <=85/102 on two checks 30 seconds apart, GPU child `nice 19`, OMP/MKL threads 4, RSS <6 GiB, MemAvailable >=16 GiB, and launch disk usage <95.5%. Existing H82 disk and retry safeguards apply; delete nothing and do not control other jobs. H93's live-ticket prerequisite is checked again by any post-merge FIFO waiter.

The source PR contains only this preregistration, the CPU-prepared runner, targeted tests, CPU dry-run support and the wave-index rows; it contains no outcomes, checkpoints, S3 payloads or large files. Local acceptance: `pytest -q research/flywheel_selfplay/h97_reg_ema/test_h97.py`, `ruff check` on owned Python files, `python -m py_compile` on owned Python files, `run_wave.py --cpu-dry-run`, and `git diff --check`. No GitHub CI is run for this PR.
