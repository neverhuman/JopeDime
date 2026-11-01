# H180 — adapter-contribution ablation and fresh-init update-budget dose

**Owner:** rd_opus [OPUS]. **Assignment:** `~/nudges/hwave_t0530_rd_opus.md`. **Claim:**
`~/jope_plan/ns/receipts/H180_claim.txt`. **Lane:** `~/JopeDime/.agent/worktrees/h180-opus`, under the H-wave
exemption at `~/jope_plan/DECISIONS_20261029.md:73`.

This is a diagnostic only. It cannot advance a model, a serving release or a live flywheel checkpoint. This file is
committed and merged before any scored operation.

## Question

H176 (verdict corrected to `DATA_SATURATED` by #1505/#1512) found that fresh native-random adapters trained for
only 40 updates score about the same as r882 on held-out families. That raises the possibility that the 29,211-parameter
adapter contributes little beyond the frozen DOPE anchor path, so that the plateau is a representation ceiling
rather than an optimizer problem.

H180 measures this directly. How much utility does the trained adapter add over the anchor alone, and does a
4× larger update budget move a fresh adapter? H179 (fresh deep ensemble) is the complementary variance test and
is not duplicated here.

## Harness (merged H176, unchanged except where stated)

**Data.** The pinned 1,219-context / 205-family real-S3 TRAIN roster and the 91-context / 19-family DL-Val
roster, with cached frozen-DOPE features. The inputs and catalog-provenance closure are byte-identical to H176's
`manifest.json` and `provenance.json`.

**Folds.** Fold seed **180140**, using H176's `family_folds`; [fold_manifest.json](fold_manifest.json) pins
them. For outer fold j:
- **A** = group j is assessed;
- **B** = group (j + 1) mod 5 is reserved and **unused**;
- **FIT** = the other 123 families.

Only FIT contexts enter minibatches, and every batch is checked against A, B and DL-Val. There is **no step
selection**.

**Treatment.** Only the 47 non-anchor adapter tensors (29,211 parameters) change. DOPE `c51a56d0` and every
non-adapter tensor are checked bitwise around every update and evaluation.

The objective, forward pass and fresh-init constructor are H176's (`treatment.adapter_forward` and
`runner.random_adapter_state`):
- AdamW, lr 1e-4, wd 1e-4, betas (0.9, 0.999);
- clip 1.0;
- SmoothL1 + 0.1·KL;
- the H156 sampler: 8 FIT contexts per update; per context, the 4 lowest-fitness candidates plus 20 others.

## Arms (one cell per outer fold × seed; seeds 180101, 180102, 180103)

| Arm | Definition |
|---|---|
| **R882** | The reference policy (Elo 1500): native r882, scored once on the TRAIN roster. |
| **ZERO_ADAPTER** | The adapter contribution removed: **score = `anchor_total`**. See the mechanism below. |
| **RAND0** | A fresh native-random adapter init with 0 updates. The init seed is `stream_seed(seed, outer, "fresh-native-init-v1")`, as in H176. |
| **FRESH40** | The same init after **40** updates. It is the step-40 checkpoint of the FRESH160 trajectory, **not** a separate run. |
| **FRESH160** | The same init and optimizer continued to **160** updates on the same batch plan (`stream_seed(seed, outer, "fit-123", "h156-batches-v1")`). |

### ZERO_ADAPTER mechanism (prespecified)

The live `FlywheelBlockFormerAdapterNet.predict_costs` computes
`total_cost = anchor_total + adapter_total_delta`.
- `anchor_total` is `anchor.predict_costs(...)["total_cost"]`, called with prior correction, struct kernel and
  cross-attention on and UCB off.
- `adapter_total_delta = mm_prior_adj + blend·sem_adj + traj_attn_adj` is the entire adapter contribution.

**ZERO_ADAPTER scores every candidate with `anchor_total`.** Zeroing the adapter's weights would *not* do this,
because `blend = 0.45·sigmoid(0) > 0` keeps a kernel term alive.

**Proof that this is the live anchor-only scorer.** For every scored ZERO_ADAPTER batch (all 91 DL-Val contexts
and every A context), the run calls the live `predict_costs`. It requires
`total_cost == anchor_total + adapter_total_delta` **bit for bit**, where `anchor_total` comes from H176's
`Engine.anchored()` with the same flags. Otherwise it stops.

The proved context count is reported. A data-free test checks the scorer and its refusal cases.

## Order of operations

1. **Regression first:** the native r882 91-context DL-Val utilities must match the pinned reference **bit for
   bit**, with mean exactly `1.1377941898164525`, using H176's native/cached identity proof.
2. Then TRAIN is scored with r882, the zero-update gap is reported descriptively, and the ZERO_ADAPTER
   decomposition proof runs on DL-Val.
3. **Timing pilot.** The pilot is fold 0, seed 180101: the first **4** FRESH160 updates. They count toward the
   trajectory, and only elapsed time is used, from the median of updates 2–4.
   - Use 5 outer folds if `5 × 3 × 160 × s/update + 1,800 s ≤ 3 h`.
   - Otherwise use **3 folds** (A = groups 0–2), disclosed, if that projection fits the 4 h hard limit.
   - Otherwise the attempt is `INCOMPLETE`.
4. **Budget.** 5 folds give **2,400 updates** and 15 fresh trajectories; 3 folds give 1,440.

## Outcomes (on each fold's untouched A families, pooled across folds)

For each arm:
- the equal-family utility per seed and the **median over seeds**;
- a **10,000-draw paired family-bootstrap 95% CI** (seed 180141). Each draw resamples A families once, shared
  across arms and seeds; computes each seed's equal-family mean; then takes the **median over seeds (axis 0, as in
  H176 #1512)**.
- the **gain against R882** with the same paired method, per-seed gains and a one-sided p = (1 + #{draw ≤ 0}) /
  10,001;
- **Elo against r882 = 1500** using the canonical H148/H176 Bradley–Terry method: context-pairwise wins with tie
  margin 1e-4, equal family weights, a 0.01 prior, 400 iterations and 10,000-draw CIs;
- the number of exact −2.0 picks;
- the **choice-agreement rate with R882** (the share of A contexts with the same argmin candidate).

Also reported: **FRESH160 − FRESH40** by the same paired method, and RAND0 and FRESH drift.

R882 and ZERO_ADAPTER are deterministic, so their three seed panels are identical. This is reported, and their
"3/3 seeds" carry no independent evidence. AUC and speed are not reported.

## Verdicts (prespecified)

**Adapter contribution** (ZERO_ADAPTER − R882):
- `ADAPTER_INERT` if the 95% CI includes 0 **and** |median gain| < 0.005;
- `ADAPTER_MATTERS` if the upper bound is < 0 (the adapter helps);
- `ADAPTER_HARMFUL` if the lower bound is > 0;
- otherwise `ADAPTER_INCONCLUSIVE`.

**Update-budget dose** (FRESH160 − FRESH40):
- `BUDGET_HELPS` if the lower bound is > 0;
- `BUDGET_HURTS` if the upper bound is < 0;
- otherwise `BUDGET_FLAT`.

**ADVANCE_CANDIDATE.** Holm with m = 4 over {ZERO_ADAPTER, RAND0, FRESH40, FRESH160} against R882 on the
one-sided p. An arm is a candidate only if:
- its median gain is **> +0.001** (gains within 1e-12 of the margin are ties and reject);
- its lower bound is > 0;
- **3/3** seeds are positive;
- its Holm-adjusted p is < 0.05.

A candidate is written up and flagged in AGENT_CHAT, **never adopted**.

**Next step.**
- `ADAPTER_INERT`: stop optimizer and regularizer adapter waves and pivot to adapter inputs and representation.
- Otherwise: continue.

## Inputs, custody and guards

**Inputs and reads.**
- Pinned local copies only: H170's r882 copy and reference, the H50 packet, the H58 feature cache and the S3
  catalog for provenance.
- Real S3 TRAIN only. DL-Val is descriptive, plus the regression gate.
- **Never** sealed, holdout or meta-test data. The audit hook refuses protected and unlisted paths.
- No live flywheel reads or writes.

**Host and limits.**
- CPU-only on xbabe2, `nice -n 19`, 1 math thread, with at most 4 aggregate process-tree threads.
- Aggregate RSS < 6 GB.
- **Stop** (no pause or resume) on any of:
  - load > 85% of cores, or host CPU > 85%;
  - MemAvailable < 16 GiB (swap is full);
  - free disk < 40 GB, or visible disk use ≥ **93.8%**;
  - 4 h of wall time.
- No GPU, no `qualification.lock`, and never xbabe0 or xbabe3.

**Failure.** Any guard or parity failure makes the attempt `INCOMPLETE`, with no scientific retry.

**Outputs.** Panels, choices, cell traces and receipts go to S3 custody, create-only with a manifest. Git carries
only code, this protocol, compact metrics (< 100 KB) and the index row.

## Flow

1. A preregistration PR with this file, the runner, monitor, analysis and treatment, and data-free tests covering:
   - zero-adapter = anchor-only scorer;
   - FRESH40 as a prefix of FRESH160;
   - a non-degenerate axis-0 bootstrap;
   - no leak from A;
   - verdict boundaries;
   - the timing rule;
   - the guards.
2. One rd_argon cross-review, with no more than 15 minutes of blocking. Otherwise `launch.json` records an
   explicit review waiver.
3. Targeted local checks.
4. Squash merge.
5. A single monitored attempt from the verified merged source.
6. A closeout PR, then removal of the lane.
