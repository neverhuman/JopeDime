# H195 — prospective confirmation of H129 BLEND_025 on fresh folds (kernel-active r882), 0 updates

**Owner:** rd_opus [OPUS]. **Assignment:** `~/nudges/hwave_t1025_h195_rdopus.md`. **Claim:**
`~/jope_plan/ns/receipts/H195_claim.txt`. **Lane:** `~/JopeDime/.agent/worktrees/h195-opus`, under the H-wave
exemption at DECISIONS line 73.

This is a diagnostic only. It cannot advance a model, a serving release or a live flywheel checkpoint. A passing
result is reported as a **CANDIDATE** for the `flywheel_gp_fused` owner to consider, **never adopted** here. This file
is merged before any outcome is opened.

## Question

H129 found **BLEND_025 = r882 + 0.25·(GBM − r882)** (cost space, calibrated HistGBM prior) a CANDIDATE against r882:
U4 gain CI95 [+0.0065, +0.0273], median seed Elo 1523. Its three seeds shared one fold partition (H124 fold seed
12400). H134, the prospective confirmation, closed INCOMPLETE on xbabe2's disk guard with no readout, and H135 asks a
different question (the flywheel's own DL-Val metric). **H195 asks again, prospectively: on families the GBM never saw,
under three independent fold partitions, does BLEND_025 beat r882 on U4?**

## Premise correction: H129 was already kernel-active

The assignment describes H129 as pre-kernel and asks for a kernel-era confirmation. **That premise is wrong.** H129's
r882 reference is the H124 `baseline`, which calls the live `adapter_10tok.predict_costs` with slots 0..3 revealed
(`revealed_indices = arange(4)`, their fitness and gate). The revealed-outcome kernel therefore already shapes every
H129 cost vector; H184 measured that kernel term as 99.96% of the adapter's effect. So:
- **BLEND_025 (primary)** is the exact H129 specification, which is kernel-active.
- **BLEND_025_NOKERNEL (descriptive)** applies the same blend to r882 costs with the KERNEL summand removed:
  `anchor_total + PRIOR + ATTN`, using the merged H184 decomposition. The runner proves on **every** TRAIN and DL-Val
  context that `anchor_total + PRIOR + KERNEL + ATTN` equals the H124 baseline cost **bit for bit** before using it
  (outcome-free smoke on 3 TRAIN contexts: PASS). **R882_NOKERNEL** (the KERNEL-removed costs alone) is reported
  alongside it as that arm's reference.

## Arms (λ fixed at 0.25; no λ search)

| Arm | Cost vector | Role |
|---|---|---|
| R882 | H124 baseline (live, kernel-active), 0 updates | reference; DL-Val native 91/91 bit-exact first |
| **BLEND_025** | `r882 + 0.25·(GBM_cal − r882)` | **primary (Holm m = 1)** |
| BLEND_025_NOKERNEL | `r882_nokernel + 0.25·(GBM_cal − r882_nokernel)` | descriptive |
| BLEND_025_SHUF | as BLEND_025, with the GBM fit on within-family permuted FIT targets | descriptive placebo |
| R882_NOKERNEL | `anchor_total + PRIOR + ATTN` | descriptive |

## Endpoint (as H129)

**H119 Track-2 U4**: the maximum utility over revealed slots 0..3 and the arm's four best-scored unrevealed
candidates (`h124.endpoints(...)["u4"]`; utility = −fitness when valid, −2.0 otherwise). Native argmin utility
(`legacy`) and exact **−2.0 pick counts** are descriptive.

## Design

**Seeds and folds.** Seeds **195101–195103**, each with its own fold seed (**195141–195143**). Each fold seed gives a
5-way family partition of the 205 DL-Train families (1,219 contexts), using the H124 `folds` algorithm.
[fold_manifest.json](fold_manifest.json) pins all three. Pairwise best-match fold Jaccard is 0.15–0.16, which is
chance level, so the partitions are independent. A test and the runner both check the partitions differ.

**Cells.** For outer fold k: **A = k, B = (k+1) mod 5, FIT = the other three**. Each family is in A once per seed.

**GBM (H129 recipe, unchanged).**
- HistGradientBoostingRegressor with `max_iter=300, learning_rate=0.08, max_leaf_nodes=63, l2_regularization=1.0`
  and `random_state = seed`.
- Features: 1,089 (the 412-d program plus `lm48_z` and `dope629`).
- Training rows: 192 unrevealed programs per context, sampled with seed **195140**; target `u − mean(u)`.
- The HistGBM recipe is H124's `fit_one`, with H195's exclusion sets.

**Fits.** Per seed and model type:
- **5 outer models**, which exclude {A, B} and are trained on FIT only;
- **10 inner models**, which exclude {A, B, j} for each FIT fold j and are used only for calibration.

The design admits every triple, because each triple of a 5-cycle contains an adjacent pair.

**Calibration.** H124's positive-slope calibration is fit per cell on the inner models' out-of-fold predictions for
the FIT contexts (slots 4.., gate-valid). Neither A nor B touches any model or calibration that scores A.

**SHUF placebo.** The training targets are permuted within each family, across all sampled rows of that family's FIT
contexts. The stream seed is `sha256(195160, seed, excluded)`. Calibration uses the same inner-model recipe.

**Budget.** At about 2 minutes per fit (H124 timing, 4 threads), 90 fits would exceed the 4 h cap. **SHUF therefore
runs on the first seed (195101) only**, giving 45 BLEND fits plus 15 SHUF fits, 60 in total.

**Stage 1 lock (written before any A or DL-Val scoring).** `stage1_lock.json` is written exclusively and fsynced
before any A or DL-Val scoring. It holds:
- every cell's calibration;
- the prediction SHA-256 of every fit it uses;
- the SHA-256 of the cell's BLEND_025 B-panel U4 vector.

Nothing is selected on B: λ is fixed and there is no search. The lock is an integrity record, and B is held out of
every model that scores A.

**Stage 2 (pooled A).** Each context is scored once per seed, using its own cell's outer model and calibration.

## Statistics and decision

For each arm against R882 on pooled-A U4:
- per-seed equal-family gain and the median over seeds;
- a **10,000-draw paired family bootstrap** (boot seed **195150**), with the same family draws for every seed,
  reduced over seeds on axis 0;
- CI95 and the H129 one-sided p against the **+0.001 margin**;
- positive seeds;
- the harm rate (families whose three-seed mean delta is < −0.01);
- canonical Bradley–Terry **Elo against r882** (400 iterations, prior 0.01, tie margin 1e-4, family-mean win
  matrices, 1,000 draws, seed 195151);
- exact −2.0 counts.

The SHUF contrast is one-seed and descriptive.

**Decision (primary only, Holm m = 1).** The result is **CANDIDATE: BLEND_025** if and only if all three hold:
1. **LCB95 > 0**;
2. **p < 0.05**;
3. **harm ≤ 10%**.

Anything else is **NO_ADVANCE**. Ties are rejected.

**DL-Val (91 contexts) is descriptive.**
- First, **R882 native 91/91 must equal the pinned contract bit-exact** (mean 1.1377941898164525), using the H152
  path. Otherwise the run stops.
- Then every arm is read on DL-Val: U4, native argmin and −2 counts, with each GBM prior being the mean calibrated
  outer-model prediction over its cells.

## Read boundary and frozen state

- **Reads:** the real-S3-derived DL-Train packet (1,219 contexts, 205 families) and the 91 DL-Val contexts (for
  regression and descriptive use only).
- **Never read:** sealed, holdout or meta-test payloads, or the live flywheel.
- The TRAIN cache reader decodes 0 DL-Val rows (H124 `train_context` with DL-Val forbidden). The DL-Val path reads the
  full cache closure, as H152 does.
- **DOPE stays `c51a56d0`, checked before and after.** The r882 checkpoint SHA (`0084479a…`) is checked before and
  after. There are 0 optimizer updates.

## Host, resources and the single-resume rule

**Host.** Inputs are staged at `~/jope-data/h195/inputs` (pinned bytes; a symlink to the host's verified copy or a
real copy). Staged hashes are identical on xbabe1 and xbabe2 (2,628 files).
- **xbabe1 is preferred**; its guard is free ≥ 40 GB.
- xbabe2 is the fallback; its guard is used < 93.8% and free ≥ 40 GB.

**Guards.** The monitor and the runner check, before every fit and stage:
- nice 19;
- OMP_NUM_THREADS=4, with every other pool at 1;
- worker-tree CPU ≤ 4.0 cores (10 s window). OS threads are recorded but not capped, because OpenMP helper threads
  exceed the compute threads;
- RSS < 6 GB;
- MemAvailable ≥ 16 GiB;
- load and host CPU ≤ 85%;
- own output root < 500 MB;
- a 4 h deadline.

**Single resume.** If and only if the monitor stops the attempt on a **resource guard** (not the deadline, and not an
exception), **one** `--resume` on the same root is allowed.
- It writes `resume-identity.json` and re-runs the regression and baselines into `*-resume` directories.
- It reuses fits whose receipt hash matches; partial fits are renamed `*.incomplete`, never deleted.
- It writes `stage1_lock-resume.json`.

A second stop closes H195 **INCOMPLETE** with no verdict. Any other failure consumes the attempt.

## Custody

The run root (fits, panels, locks and logs) goes create-only to
`s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H195/<run_id>/`, with a manifest. The
closeout commits `metrics.json` (< 100 KB), a README and an index row.
