# H184 — which part of the adapter's structure carries the training-free lift?

**Owner:** rd_opus [OPUS]. **Assignment:** `~/nudges/hwave_t0645_h184_opus.md`. **Claim:**
`~/jope_plan/ns/receipts/H184_claim.txt`. **Lane:** `~/JopeDime/.agent/worktrees/h184-opus`, under the H-wave
exemption at DECISIONS line 73.

This is a diagnostic only. It cannot advance a model, a serving release or a live flywheel checkpoint. This file is
committed and merged **before any A outcome is opened**.

## Question

H180 found that a fresh *random*, untrained adapter recovers the whole adapter benefit over the anchor alone. As a
post-hoc result, RAND0 − ZERO_ADAPTER was +0.060 [+0.035, +0.092], training added nothing detectable, and
ZERO_ADAPTER made 35 invalid (−2.0) picks against r882's 17.

If the lift is structural, the next lift comes from the structure. H184 asks **which of the adapter's output terms
carries it**, using only 0-update random weights.

H183 (gpu_fused_redteam_codex) covers ensembles and the untrained-ensemble RENS10 control. H184 does not duplicate
it.

## Structure under test

The live `FlywheelBlockFormerAdapterNet.compute_adapter_delta_from_static_terms` returns
`adapter_total_delta = PRIOR + KERNEL + ATTN`, summed in that order:

| Term | Live expression | What it is |
|---|---|---|
| **PRIOR** | `mm_prior_adj` | the multimodal BlockFormer prior correction from the random encoders |
| **KERNEL** | `0.45·sigmoid(traj_kernel_blend) · sem_adj` | a Nadaraya–Watson regression of the revealed candidates' residuals on joint semantic/embedding distance, with a null weight |
| **ATTN** | `tanh(traj_attn_scale) · centered traj_delta_head(...)` | null-token trajectory cross-attention over the revealed history |

**Structural facts of the native constructor**, read before any outcome:
- the kernel's scalar parameters (`sem_kernel_scale_raw`, `emb_kernel_scale_raw`, `null_kernel_w0_raw` and
  `traj_kernel_blend`) are **fixed constants** at init, not random;
- the attention head's last layer is initialized with std 1e-3 and zero bias.

So under "random weights", KERNEL's randomness enters only through the random encoders (`sem24` and `cand_q`), and
ATTN starts small.

**Decomposition.** `treatment.adapter_components` replays the live function operation for operation and returns
the three terms. **On every scored batch the run requires** both of the following, and otherwise stops:
- `PRIOR + KERNEL + ATTN` equals the live `compute_adapter_delta_from_static_terms` output **bit for bit**;
- RAND0 equals H180/H176's `adapter_forward` score bit for bit.

A data-free test also checks the bit-exact sum against the pinned live method on a synthetic adapter with the
live shapes and init rules.

## Arms (6; all 0 optimizer updates)

| Arm | Score |
|---|---|
| R882 | native r882 (the reference, Elo 1500) |
| ZERO_ADAPTER | `anchor_total`, as in H180, with the live-decomposition proof |
| **RAND0** | `anchor_total + (PRIOR + KERNEL + ATTN)` under the cell's fresh native random init |
| RAND0_NO_PRIOR | `anchor_total + (KERNEL + ATTN)`, using the same init |
| RAND0_NO_KERNEL | `anchor_total + (PRIOR + ATTN)`, using the same init |
| RAND0_NO_ATTN | `anchor_total + (PRIOR + KERNEL)`, using the same init |

- **The ablations act on the output.** Each drops one term from the final score and keeps the internal pathways.
  For example, PRIOR still enters `eff_prior` and the kernel residuals in RAND0_NO_PRIOR. An ablation measures the
  term's *direct* contribution given the others.
- **Init.** Each (outer fold, seed) uses one fresh native init, `stream_seed(seed, outer, "fresh-native-init-v1")`
  as in H180, shared by RAND0 and all three ablations, so the comparisons are paired.

## Harness

**Data.** The merged H180/H176 harness, with the pinned 1,219-context / 205-family real-S3 TRAIN roster and the
cached frozen-DOPE features.

**Folds.** **Fresh fold seed 184140** ([fold_manifest.json](fold_manifest.json)). Because nothing is trained,
FIT and B are unused. Each outer fold's A (41 families) is scored with that cell's random init, so every family is
assessed once per seed, under seeds **184101–184103**.

**Integrity.**
- DOPE `c51a56d0` and every non-adapter tensor are checked bitwise around every evaluation.
- The runner asserts that **0 optimizer updates** occurred and that the r882 adapter state was restored.

## Order of operations

1. **Regression first:** the exact native r882 91-context DL-Val utilities, bit for bit, with mean
   `1.1377941898164525`.
2. Then r882 is scored on TRAIN, and the zero-update gap is reported descriptively.
3. Then the ZERO_ADAPTER decomposition proof runs on DL-Val.
4. Then the 15 cells run. There is no training and no timing pilot.

## Outcomes (pooled A panel: 205 families and 1,219 contexts per seed)

For each arm:
- the median over seeds of the equal-family utility, with a 95% CI;
- the **gain against R882** — median of the 3 seeds' equal-family contrasts, with a **10,000-draw paired family
  bootstrap (seed 184141)**, families resampled once per draw and shared across arms and seeds, then the median
  over seeds (**axis 0**, as in #1512);
- per-seed gains and a one-sided p;
- **Elo against r882 = 1500** using the canonical H148/H176 Bradley–Terry method with 10,000-draw CIs;
- the **exact −2.0 counts**;
- choice agreement with R882.

R882 and ZERO_ADAPTER are deterministic, so their identical seed panels are disclosed.

## Decisions (prespecified)

**Attribution (primary).** For each term X, the lift X carries is `RAND0 − RAND0_NO_X`, by the same paired
method, with **Holm m = 3** over {PRIOR, KERNEL, ATTN} on the one-sided p.
- `CARRIES_LIFT` if the 95% lower bound is > 0 **and** the Holm-adjusted p is < 0.05.
- `HURTS` if the upper bound is < 0.
- Otherwise `NEUTRAL`.

The overall label is `CARRIERS:<X+…>`, or `NO_SINGLE_CARRIER` if none qualifies. Each term's
**share of RAND0 − ZERO_ADAPTER** is reported descriptively; the terms interact, so shares need not sum to 1.

**ADVANCE_CANDIDATE** (strict, unchanged): **Holm m = 4** over {RAND0, RAND0_NO_PRIOR, RAND0_NO_KERNEL,
RAND0_NO_ATTN} against R882. An arm is a candidate only if:
- its median gain is **> +0.001** (gains within 1e-12 of the margin are ties and reject);
- its lower bound is > 0;
- **3/3** seeds are positive;
- its Holm-adjusted p is < 0.05.

A candidate is written up and flagged to the flywheel owner, **never adopted**. ZERO_ADAPTER is a control and is
not eligible.

## Inputs, custody and guards

**Inputs and reads.**
- Pinned local copies only, identical to H180's manifest: the H170 r882 copy and reference, the H50 packet, the
  H58 cache and the catalog provenance.
- Real S3 TRAIN only. DL-Val serves only the regression gate and the descriptive gap.
- **Never** sealed, holdout or meta-test data; the audit hook refuses protected and unlisted paths.
- The live flywheel is never touched.

**Host and limits.**
- CPU-only on xbabe2 at `nice -n 19`, 1 math thread, with at most 4 aggregate threads.
- RSS < 6 GB.
- **Stop** on any of:
  - load > 85% of cores, or host CPU > 85%;
  - MemAvailable < 16 GiB;
  - free disk < 40 GB, or use ≥ **93.8%**;
  - 4 h of wall time.
- No GPU, and never xbabe0 or xbabe3.

**Failure.** Any guard or parity failure makes the attempt `INCOMPLETE`, with no scientific retry.

**Outputs.** Panels, choices and receipts go to create-only S3 custody with a manifest. Git carries code, this
protocol, metrics (< 100 KB) and the index row.

## Flow

1. A preregistration PR with this file, the code and data-free tests covering:
   - the decomposition matching the live delta bit for bit;
   - ablation semantics and refusals;
   - the attribution and Holm logic;
   - the axis-0 bootstrap;
   - the ADVANCE gates, including the margin tie;
   - Elo 1500 for identical arms;
   - the guards.
2. rd_argon's cross-review.
3. Targeted local checks.
4. Squash merge.
5. A single monitored attempt from the verified merged source.
6. A closeout PR, then removal of the lane.
