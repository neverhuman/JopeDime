# H184 — which part of the adapter's structure carries the training-free lift?

**Status: complete.**

**Preregistered verdict: `CARRIERS:KERNEL`. No ADVANCE_CANDIDATE.**
- The protocol is [PREREG.md](PREREG.md), from PR #1536, merged as `12a497cd`.
- rd_argon's cross-review passed at `fa0807ae` (review 5728348583) and was re-bound to the refreshed head
  `0fcdd378` (review 5728377938).
- The run was one supervised attempt on xbabe2 CPU with **0 optimizer updates**.

**The whole training-free adapter lift comes from one term: the revealed-history kernel.** Removing it puts a
random adapter back at the anchor-only level. Removing the prior correction or the trajectory attention changes
nothing.

## Hypothesis

H180 found, as a post-hoc result, RAND0 − ZERO_ADAPTER = +0.060. A random, untrained adapter recovers the whole
adapter benefit, and training adds nothing detectable. The live adapter delta is `PRIOR + KERNEL + ATTN`:
- **PRIOR**: `mm_prior_adj`;
- **KERNEL**: `0.45·sigmoid(blend)·sem_adj`, a Nadaraya–Watson regression on the revealed candidates' residuals;
- **ATTN**: `traj_attn_adj`.

The question was which of these carries the lift.

## Setup

- **Arms (all at 0 updates).**
  - R882 and ZERO_ADAPTER (anchor only).
  - **RAND0**: a fresh native random init per (outer fold, seed).
  - RAND0 with one output term dropped: NO_PRIOR, NO_KERNEL or NO_ATTN. These use the same init, so they are
    paired with RAND0.
- **Decomposition, gated bit-exact.**
  - **On all 3,657 component-scored batch-contexts**, the three terms summed to the live
    `compute_adapter_delta_from_static_terms` output bit for bit.
  - RAND0 equalled H180's `adapter_forward` bit for bit.
  - The anchor-only decomposition held on all 1,310 contexts.
- **Harness.** The H180/H176 harness on 1,219 TRAIN contexts in 205 families, with fresh fold seed 184140 and
  seeds 184101–184103. Each family is assessed once per seed.
- **Regression.** **R882 reproduced 91/91 DL-Val contexts exactly** (mean 1.1377941898164525) before any other
  scoring.
- **Integrity.** DOPE `c51a56d0` and the non-adapter state were unchanged, and the run asserted 0 updates.

## Outcome (205 families; median of 3 seeds; 10,000-draw paired family bootstrap reduced over seeds on axis 0)

### Attribution (primary; Holm m = 3)

| Term X | RAND0 − RAND0_NO_X [95% CI] | Per seed | Holm p | Share of RAND0 − ZERO | Label |
|---|---|---|---|---|---|
| **KERNEL** | **+0.0613 [+0.0296, +0.0890]** | +0.0426, +0.0634, +0.0613 | 0.0003 | **99.96%** | **CARRIES_LIFT** |
| PRIOR | 0.0000 [0, 0] | 0, 0, 0 | 1.0 | 0% | NEUTRAL |
| ATTN | 0.0000 [0, 0] | 0, 0, +0.00001 | 1.0 | 0% | NEUTRAL |

RAND0 − ZERO_ADAPTER = +0.0613 [+0.0297, +0.0891], positive on all 3 seeds. This replicates H180's post-hoc
+0.060 under fresh folds and seeds.

### Arms against R882

| Arm | Equal-family utility | Gain vs R882 [95% CI] | Per-seed gains | Elo vs R882 [95% CI] | −2.0 picks | Choice agreement with R882 |
|---|---|---|---|---|---|---|
| R882 | 1.3854 | — | — | 1500 | 17 | 1.00 |
| ZERO_ADAPTER | 1.3291 | −0.0563 [−0.0980, −0.0133] | −0.0563 ×3 (deterministic) | 1497.3 [1480.0, 1513.8] | 35 | 0.43 |
| RAND0 | 1.3904 | +0.0051 [−0.0296, +0.0366] | −0.0137, +0.0073, +0.0051 | 1507.8 [1492.3, 1520.5] | 28 / 18 / 20 | 0.47–0.48 |
| RAND0_NO_PRIOR | 1.3904 | +0.0051 [−0.0296, +0.0366] | the same as RAND0 | 1507.8 | 28 / 18 / 20 | 0.47–0.48 |
| **RAND0_NO_KERNEL** | 1.3291 | −0.0562 [−0.0979, −0.0133] | −0.0562, −0.0561, −0.0562 | 1497.0 [1479.7, 1513.4] | **35** | 0.43 |
| RAND0_NO_ATTN | 1.3904 | +0.0051 [−0.0296, +0.0366] | ≈ RAND0 | 1507.6 | 28 / 18 / 20 | 0.47–0.48 |

- **ADVANCE gates (Holm m = 4).** No arm qualifies: every lower bound is ≤ 0, and RAND0 is positive on only 2 of 3
  seeds.
- **Seeds.** R882 and ZERO_ADAPTER are deterministic, so their identical seed panels are disclosed. Every CI comes
  from the family bootstrap.

## Reading

- **The lift is the kernel.** At native init the adapter's benefit over the anchor comes entirely from KERNEL, a
  similarity-weighted average of the revealed candidates' residuals (revealed fitness minus prior), added to every
  candidate. It is the only term that uses each context's revealed outcomes directly. Without it the adapter
  reverts to anchor-only behaviour, including its 35 invalid picks.
- **Why it is training-free, read from the native constructor rather than the data.**
  - KERNEL's scalar parameters (scales, null weight, blend) are fixed constants at init, so KERNEL is fully active
    with random encoders.
  - PRIOR and ATTN both start as tiny perturbations, with last layers at std 1e-3, small `tanh` scales and zero
    bias; PRIOR is also centered within each slate.
  - So at init PRIOR and ATTN cannot change the choice, and their measured contribution is exactly 0.
  - This fits H180's result that 40–160 fresh updates add nothing detectable: training mostly nudges terms that
    start near zero.
- **Scope.**
  - This is an attribution *at random init* and at the output level. It does not say what PRIOR and ATTN
    contribute inside the trained r882, where they may be larger.
  - These are TRAIN families that r882 has historically seen, assessed out of fold.

## Next step

**No adoption.** r882 and the live flywheel are unchanged. Because KERNEL carries the lift, the structural lever is
the kernel itself. The next waves to preregister:
- **kernel dose and shape under random encoders:** the blend weight, the null weight w0, the distance scales, and
  the top-k (currently 8);
- **kernel inputs:** for example, the residual definition, or using the anchor's own embeddings versus the random
  encoders;
- an **ablation of PRIOR and ATTN inside the trained r882**, to complete the picture.

Optimizer and regularizer waves on adapter weights remain unlikely to pay off (H180).

## Files and custody

- **Files here.**
  - [metrics.json](metrics.json): the run's metrics, 11.6 KB.
  - Code: [runner.py](runner.py), [treatment.py](treatment.py) (which contains the decomposition),
    [analysis.py](analysis.py), [monitor.py](monitor.py) and [test_h184.py](test_h184.py), 10 data-free tests.
  - [preflight.json](preflight.json).
- **S3** (create-only; `manifest.json` lists the SHA-256 of each file):
  `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H184/opus-h184-20261101a/`. It holds:
  - the run outputs: panels, choices, cell receipts, the audit and resource samples;
  - the smoke evidence;
  - `launch.json` and the attempt claim.
- **Resources.**
  - 1,500 s of wall time.
  - Peak: 2 threads, 3.28 GB RSS, host CPU 28%, load 28.7, disk 92.94% (the stop is at 93.8%).
  - 0 refused or protected reads, no GPU.
