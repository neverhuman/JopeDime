# H192 — strict-margin (confidence-gated) adapter selection, 0 updates

**Owner:** rd_opus [OPUS]. **Assignment:** `~/nudges/hwave_t0900_h192_opus.md`. **Claim:**
`~/jope_plan/ns/receipts/H192_claim.txt`. **Lane:** `~/JopeDime/.agent/worktrees/h192-opus`, under the H-wave
exemption at DECISIONS line 73.

This is a diagnostic only. It cannot advance a model, a serving release or a live flywheel checkpoint. On ADVANCE
it is written up and flagged to the `flywheel_gp_fused` owner, **never adopted**. This file is merged before any
outcome is opened.

## Question

H180, H184 and H188 found that the adapter's value is its revealed-outcome kernel steering picks away from
invalid (−2.0) outcomes, not learned weights. H186 (kernel scale) and H188 (kernel inputs) did not clear the gate.
The untested lever is **when** to trust the adapter's re-ranking: override the base pick only when the adapter is
confidently better. H190 (ENS10) and H191 (kernel-scale KX) are not touched.

## Definitions (exact, from the code)

For every context, with r882 at its pinned weights and 0 updates:

| Term | Definition |
|---|---|
| **Base pick** | `argmin anchor_total`: r882 with the **whole adapter re-rank removed**. This is the H180 anchor-only scorer, proved bit for bit against the live `total_cost = anchor_total + adapter_total_delta` on every scored context. |
| **Full pick** | `argmin(anchor_total + adapter_total_delta)`: r882's own pick (R882). |
| **Gate statistic** | `margin = (full[base pick] − full[full pick]) / std(full over the context's pool)`. Here `full` is the r882 cost vector, so the margin is ≥ 0. It is 0 when the two picks agree or the std is 0. |
| **Gated pick at τ** | **τ = 0 is R882 exactly**: the full argmin, including its first-index tie-break. For τ > 0, the full pick is used if `margin > τ` (strictly); otherwise the base pick. |
| **τ grid** | {0, q25, q50, q75, q90}, where the quantiles are of `margin` over the **FIT contexts of that cell where the two picks disagree** (margin > 0), using linear interpolation. Agreeing contexts are excluded, because they would put most quantiles at 0. |

**Override rate** is the share of A contexts with disagreeing picks where the full pick is used, reported per τ.

## Design

**Folds.** The **three seeds 192101–192103 use different fold seeds**, 192141, 192142 and 192143. Each gives a fresh
cyclic 3/1/1 split of the 205 DL-Train families (1,219 contexts) using H176/H188 `family_folds`;
[fold_manifest.json](fold_manifest.json) pins them. The runner and a test require the three partitions to differ.

**Stage 1 (selection; no A or DL-Val input).** For each (seed, outer fold k):
- build the τ grid from FIT_k;
- pick **τ\* = argmax of the equal-family utility on B_k**;
- **break ties toward the earliest grid entry, i.e. τ = 0, preferring R882**.

All 15 cells (τ grid, τ\*, B utilities, B-panel SHA-256s) are written to `mgate_selection_lock.json` and
**fsynced before any A or DL-Val readout**. A test confirms the lock is unchanged when A utilities are perturbed.

**Stage 2 (pooled A).** **MGATE** is each cell's τ\* applied to its A_k. Every family is scored once per seed.
**MGATE against R882 is the one primary contrast (Holm m = 1).** Fixed-τ pooled readouts are descriptive only.

**Reported:**
- per-seed gains;
- the **median over seeds** with a **10,000-draw paired family bootstrap (boot seed 192150), reduced over seeds on
  axis 0**;
- the H177 pooled-selector decision: seed-averaged family gains, gain > +0.001 + 1e-12, lower bound > 0, p < 0.05
  (m = 1);
- **Elo against r882** (canonical Bradley–Terry, context-level wins, tie margin 1e-4) with a CI;
- the **exact −2.0 pick count**;
- the **override rate per τ**;
- **τ\* stability**, as counts per grid position across the 15 cells.

**DL-Val (91 contexts) is descriptive.** It uses the τ grid built from all 205 TRAIN families, together with R882.

**Reuse.**
- The engine, anchor-only scorer and pins come from merged H188/H184/H180/H176.
- The fold, utility, family, Holm, win-matrix and Bradley–Terry helpers are **imported** from merged H188
  `analysis.py`, which is hash-pinned.
- The pooled rule is H177/H188's with the arm count made a parameter. H177 hard-codes m = 4 and H188 asserts
  m = 3, so neither can be imported for m = 1.

**Disclosed before any outcome.** At 0 updates, every per-context utility (R882, base and full) is **identical
across seeds**; the seeds only reshuffle folds, so τ\* can differ by cell. **The family bootstrap CI is the
inference.** Per-seed counts are not independent evidence.

## Verdict (strict; ties reject)

**`ADVANCE` only if all of these hold:**
- the H177 pooled selector admits MGATE;
- the MGATE pooled-A **median gain is > +0.001**;
- its **lower bound is > 0**;
- **the H178 receipt gate passes**: `h178_receipt.json` is COMPLETE, shows null or harmful admission ≤ 0.05, and
  binds the current H177 `analysis.py` bytes.

Otherwise the verdict is `NO_ADVANCE`.

## Hard rules, order and guards

**Order.**
1. **Regression first:** the native r882 91-context DL-Val utilities must match bit for bit, with mean
   `1.1377941898164525`.
2. **τ = 0 must equal R882 bit for bit** on every TRAIN context, and on DL-Val against the pinned reference.
   Otherwise the run stops.

**Integrity.** DOPE `c51a56d0` and every non-adapter tensor are checked bitwise around every evaluation. The r882
adapter hash is unchanged, with 0 updates.

**Inputs.** Pinned local copies identical to H188's. Real S3 TRAIN only. **Never** sealed, holdout or meta-test
data. The live flywheel is never touched.

**Host and limits.**
- CPU on xbabe2 at `nice -n 19`, with at most 4 aggregate threads and RSS < 6 GB.
- **Stop** on any of: load > 85% of cores or host CPU > 85%; MemAvailable < 16 GiB; free disk < 40 GB or use
  ≥ 93.8%; 4 h of wall time.
- **At claim time the disk was at 93.61%**, close to the stop. A rise from other agents' work could stop this run
  as `INCOMPLETE`. H192 itself writes only a few MB.
- No GPU, and never xbabe0 or xbabe3.

**Failure.** Any guard or parity failure makes the attempt `INCOMPLETE`, with no retry.

**Outputs.** Panels, gates, the lock and receipts go to create-only S3 custody with a manifest. Git carries code,
this protocol, metrics (< 100 KB) and the index row.
