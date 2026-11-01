# H188 — kernel-input ablation on the r882 adapter (0 updates)

**Owner:** rd_opus [OPUS]. **Assignment:** `~/nudges/hwave_t0715_rd_opus.md`. **Claim:**
`~/jope_plan/ns/receipts/H188_claim.txt`. **Lane:** `~/JopeDime/.agent/worktrees/h188-opus`, under the H-wave
exemption at DECISIONS line 73.

This is a diagnostic only. It cannot advance a model, a serving release or a live flywheel checkpoint. A candidate
is written up and flagged to the `flywheel_gp_fused` owner in AGENT_CHAT, **never adopted**. This file is merged
before any outcome is opened.

## Question

H184 found that the adapter's revealed-history kernel carries 99.96% of the training-free lift. H186 (rd_astra)
tunes the kernel's *constants*. H188 changes the kernel's *inputs* on the trained r882 adapter, one change per arm,
with no training. H187's robust loss and family balance are not touched.

## Arms (reference R882; Holm m = 3; each arm makes exactly one input change)

Every arm scores `anchor_total + (PRIOR + KERNEL' + ATTN)`, where PRIOR and ATTN are r882's own terms, unchanged,
and only the kernel term is recomputed from a changed input.

| Arm | The one change to the kernel's inputs |
|---|---|
| **KIN_ANCHOR** | The similarity features come from the **frozen anchor**, read-only, via `anchor.encode_candidates(...)`: the L2-normalized structural signature (24-d) replaces the adapter's `sem24`, and the L2-normalized anchor candidate embedding replaces `cand_q_norm`. Kernel scales and weights are unchanged. |
| **KIN_RANKRES** | The revealed residuals `r_rev` are **rank-normalized within each context** before the Nadaraya–Watson average: `r' = mean(r) + std(r)·u / std(u)`, where `u` is the average-tie rank minus (K + 1)/2. This keeps the context's own location and scale and removes outlier magnitude. If std(r) = 0, r is used unchanged. |
| **KIN_CLIPRES** | The revealed residuals are **Huber-winsorized around the context median** at `1.345 × 1.4826 × MAD`, so large residuals such as failed gates, which enter capped at 2.5, cannot dominate. If MAD = 0, r is used unchanged. |

**Gates on every scored batch** (otherwise the run stops):
- `PRIOR + KERNEL + ATTN` equals the live `compute_adapter_delta_from_static_terms` output bit for bit (H184);
- **the replayed kernel with identity inputs equals the live KERNEL term bit for bit**;
- R882 equals H176/H180's `adapter_forward` bit for bit;
- R882 on the live path equals the native r882 utility for every TRAIN context.

Data-free tests check the bit-exact identity against the pinned live method on a synthetic adapter, and the
semantics of both residual transforms.

**Grid.** No grid extension after results.

## Design (merged H177 pooled-selector rule; H178 calibration receipt pinned)

**Folds.** Each of the **3 seeds 188101–188103 is a fresh reshuffle** of the 5 family folds: H176 `family_folds`
with seed `stream_seed(188140, seed)`; see [fold_manifest.json](fold_manifest.json). Within each fold, A = g[j] and
B = g[(j+1) mod 5]. FIT is unused, since there are 0 updates.

**Stage 1 (pooled B).** Each arm is scored on every fold's B families, so per seed every one of the 205 families
is scored once. Then `pooled_b_select_m3` runs. It is merged H177's `pooled_b_select`, copied operation for
operation, with the arm set made a parameter for this m = 3 family:
- seeds are averaged within each family;
- a 10,000-draw paired family bootstrap (seed 188141);
- **admit iff the gain is > +0.001 (gains within 1e-12 of the margin are ties and reject), the 95% lower bound is
  > 0 and the Holm (m = 3) p is < 0.05**.

The decisions are **locked, with their SHA-256 recorded, before any A scoring**.

The H178 receipt (`h178_receipt.json`) calibrated this rule for H177's four-arm family, finding a null or harmful
admission rate of 0.0. Here the rule is applied with m = 3, which is disclosed.

**Stage 2 (pooled A).** Each arm is scored on every fold's A families. For each arm against R882:
- per-seed gains and the **median over seeds**, with a 10,000-draw paired family-bootstrap CI reduced over seeds
  (**axis 0**, #1512);
- the Holm (m = 3) p;
- **Elo against r882** (canonical Bradley–Terry) with a CI;
- the **exact −2.0 pick count**;
- choice agreement.

DL-Val (91 contexts) is reported descriptively.

**Verdict.** `ADVANCE:<arm>` only if the arm is **admitted on pooled B and** has a pooled-A median gain > 0
**and** 3/3 seeds positive **and** a pooled-A lower bound > 0. Ties reject. Otherwise the verdict is `NO_ADVANCE`.

### Disclosed before any outcome: B and A are not independent here

At 0 updates on fixed r882 weights, every arm's per-context utility is **deterministic**. In each reshuffle every
family appears once in pooled B and once in pooled A, so **the pooled-B and pooled-A panels and all three seed
panels contain identical values**. Consequences:
- **Stage 2 is a determinism replicate of Stage 1, not an independent confirmation.**
- "3/3 seeds" is all-or-nothing.
- The evidence is effectively **one 205-family paired test per arm**, Holm-corrected over 3.

The run records `pooled_b_equals_pooled_a` and the identical-seed flags. The owner's two-stage, three-seed design
is kept as specified so that it composes with H177/H178, and this limitation applies to its reading.

## Order, inputs and guards

**Order.**
1. **Regression first:** the native r882 91-context DL-Val utilities must match bit for bit, with mean
   `1.1377941898164525`.
2. r882 is scored on TRAIN, with the zero-update gap reported descriptively.
3. The anchor-only decomposition proof runs on DL-Val.
4. Stage 1 runs, then the lock is written, then Stage 2, then DL-Val (descriptive).

**Integrity.** DOPE `c51a56d0` and every non-adapter tensor are checked bitwise around every evaluation. The
anchor is only read. The r882 adapter state is hash-checked unchanged at the end, with 0 updates.

**Inputs.** Pinned local copies identical to H184's: the H170 r882 copy and reference, the H50 packet, the H58
cache and the catalog provenance. Real S3 TRAIN only. **Never** sealed, holdout or meta-test data; the audit hook
refuses them. The live flywheel is never touched.

**Host and limits.**
- CPU on xbabe2 at `nice -n 19`, 1 math thread, with at most 4 aggregate threads.
- RSS < 6 GB.
- **Stop** on any of: host CPU or load > 85%; MemAvailable < 16 GiB; free disk < 40 GB or use ≥ 93.8%; 4 h of wall
  time.
- No GPU, and never xbabe0 or xbabe3.

**Failure.** Any guard or parity failure makes the attempt `INCOMPLETE`, with no retry.

**Outputs.** Panels, choices, the lock and receipts go to create-only S3 custody with a manifest. Git carries code,
this protocol, metrics (< 100 KB) and the index row.
