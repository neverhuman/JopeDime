# H217 — fresh-fold confirmation of H210 GATED_025 (frozen gate), xbabe2 CPU, 0 updates

**Status: COMPLETE. Verdict: CONFIRMED: GATED_025 + NO_PLATEAU_WIN.** The frozen H210 gate keeps a native gain over
r882 on fresh fold partitions: LCB95 > 0, 3/3 seeds and harm 4.4% (cap 10%). It does not lift the flywheel91 DL-Val
plateau by the preregistered margin: +0.00145, against a required +0.002. No model, checkpoint, DOPE or serving state
changed. Under the PREREG, CONFIRMED earns a **shadow-only note** at most; it does not advance anything.

## Hypothesis

[H210](../h210_blend025_gated_retry/README.md) found CANDIDATE: GATED_025 on one set of fold partitions: native
+0.0140 [+0.0022, +0.0355], harm 3.4%, 3/3 seeds, Elo 1505.6. Its flywheel91 DL-Val gain was only +0.0010
(descriptive), and its U4 CI crossed 0. H217 asks two questions. See [PREREG.md](PREREG.md).

1. **Primary.** Does the frozen GATED_025 recipe, re-run on fresh partitions with no retuning, keep a native gain over
   r882 with harm ≤ 10% in every seed?
2. **Co-primary.** Does it lift the flywheel91 plateau: a DL-Val context-mean gain over r882 above +0.002, with a
   family-bootstrap LCB95 > 0?

## Setup

- **Prereg.** PR #1817, merged as `e8364eb7`, plus amendment A1 (PR #1824, merged as `22a3f1a3`). The run used the
  merged commit `22a3f1a3`, whose H217 blobs equal the reviewed head `7dba0f2e`. The PREREG sha256 is `7a03ef84…`.
- **Frozen recipe.** H210's blend (r882 + 0.25·(calibrated HistGBM − r882)), its `zgap` gate and its FIT-only τ
  procedure: maximize FIT native gain subject to FIT harm ≤ 5%, else fall back to r882. A test checks that the
  function bodies are identical to H210's. No τ is carried over; every τ is re-chosen on this wave's FIT folds.
- **Data.** 205 DL-Train families and 1,219 contexts, from the H124 pinned packet (real S3-derived data, staging hash
  list `2f7a3642…`, 2,628 files).
- **Seeds.** 217101–217103, with fold seeds 217141–217143. Each seed has its own partition:
  - best-match Jaccard between seeds is 0.149–0.182;
  - against earlier waves it is at most 0.183 (H210) and 0.179 (H203).
- **Cells and fits.** A = k, B = k+1, FIT = the other three. There are 45 FIT-only HistGBM fits with positive-slope
  calibration. The stage-1 lock (`a432d515…`, calibration and τ) was fsynced before any A or DL-Val scoring.
- **Primary statistic.** Pooled-A native utility (`fixed-dev-cost-argmin-negative-fitness/v1`). The bootstrap uses
  10,000 paired family-cluster draws (seed 217150), reduced over seeds on axis 0. GATED_025 is the sole primary.
- **Co-primary statistic.** The DL-Val 91-context mean (19 families) against r882's 1.1377942, with a 10,000-draw
  family-cluster bootstrap (seed 217152).
- **Frozen state.** DL-Val r882 native 91/91 was bit-exact first (1.1377941898164525). DOPE was `c51a56d0` before
  and after, with 0 updates. No GPU and no CUDA initialization.
- **Reads.** 0 sealed, holdout, meta-test or live-loop reads, and 0 DL-Val rows decoded by the TRAIN reader.

## Outcome: pooled A, 205 families, native first-argmin against R882 (Elo: r882 = 1500)

| Arm | Median gain | Per-seed | CI95 | p (vs +0.001) | Seeds > 0 | **Harm** | Elo [CI95] | −2.0 picks (R882: 17/17/17) |
|---|---:|---|---|---:|---:|---:|---|---|
| **GATED_025 (primary)** | **+0.0216** | +0.0132 / +0.0230 / +0.0216 | **[+0.0038, +0.0418]** | 0.0006 | **3/3** | **4.4%** (9) | **1504.8** [1500.9, 1510.4] | 16 / 14 / 13 |
| GATED_025_DIS (H129 disagreement placebo) | −0.0012 | −0.0018 / −0.0012 / +0.0008 | [−0.0041, +0.0007] | 0.99 | 1/3 | 7.3% (15) | 1499.0 [1496.2, 1501.8] | 17 / 17 / 17 |
| BLEND_025 (ungated component) | +0.0367 | +0.0367 / +0.0441 / +0.0294 | [+0.0022, +0.0778] | 0.020 | 3/3 | 25.9% (53) | 1515.9 [1505.2, 1527.7] | 18 / 8 / 15 |

**Primary gates for GATED_025.**
- LCB95 > 0: ✓ (+0.0038).
- 3/3 seeds positive: ✓.
- Harm ≤ 10%: ✓ (4.4%).

The primary is **CONFIRMED: GATED_025**. The p-value is reported, not gated.

**Co-primary: flywheel91 DL-Val (91 contexts, 19 families).**

| Arm | Context mean | Gain vs R882 (1.13779) | CI95 | Equal-family mean (R882 1.15362) |
|---|---:|---:|---|---:|
| **GATED_025** | **1.13924** | **+0.00145** | **[+0.00038, +0.00270]** | 1.15500 |
| GATED_025_DIS | 1.13647 | −0.00132 | [−0.00444, +0.00073] | 1.15235 |
| BLEND_025 | 1.13159 | −0.00620 | [−0.03374, +0.01580] | 1.14767 |

GATED_025's lower bound is above 0, but its gain is below the +0.002 margin. The co-primary is therefore
**NO_PLATEAU_WIN**: the PREREG says a gain at or below +0.002 is not a win. No arm made a −2.0 pick on DL-Val.

**U4 (report-only).**
- GATED_025: +0.0034 [−0.0006, +0.0099], harm 4.4%, 3/3 seeds, Elo 1514.3. The CI crosses 0, as in H210.
- BLEND_025: +0.0123 [+0.0021, +0.0216], harm 8.3%, Elo 1520.4.
- GATED_025_DIS: +0.0005, Elo 1502.5.

## Gate behaviour

- **Switching.** The blend's pick differs from r882's in 32–36% of A contexts per seed. GATED_025 opens on
  13.0% / 25.3% / 26.0% of those switching contexts, and on 55% / 59% / 77% of all contexts (an agreeing context
  counts as open but changes nothing). The placebo opens on 8.6% / 7.3% / 19.5% of switching contexts.
- **τ per cell (FIT only).** τ is 0.0023–0.0280 in 13 cells. The gate stayed fully closed (fall back to r882) in 2
  cells, `217101/0` and `217102/3`; H210 also had 2 closed cells. FIT harm is ≤ 4.9% in every cell.
- **Interpretation.**
  - The H210 result replicates on fresh partitions, with a slightly larger median (+0.0216 vs +0.0140) and similar
    harm (9 vs 7 families) and Elo (1504.8 vs 1505.6).
  - The placebo is again null with a similar open rate. So the selection by `zgap`, not merely opening less often,
    is what keeps the gain.
  - The gate keeps about 60% of the ungated median gain (+0.0216 of +0.0367) while cutting harmed families from 53
    to 9.
  - On DL-Val the gated arm is positive with LCB > 0 in both waves (+0.0010 in H210, +0.00145 here), while the
    ungated blend is negative. The effect is real but small: it is under the preregistered plateau margin.

## Shadow log (step 4, read-only)

Two xbabe1 units, `h217-shadow` and `h217-shadow-h217gate` (nice 19, every 600 s), read the live flywheel's
`round-N.json` files and write only under `~/jope-data/h217/shadow/`. For every completed round they log the live
candidate's DL-Val utility next to what GATED_025 scores on the same 91 contexts, using the frozen picks recomputed
from H210 (`gated_dlval-h210.json`) and from this run (`gated_dlval-h217.json`). The helper
`gated_dlval.py` (in custody under `shadow/`) self-checks that its R882 values equal the pinned reference and its
means equal `metrics.json`. There is no loop stop and no promotion change.

At closeout (rounds 775–3699, 2,925 rounds, 0 skipped):
- the live candidate beat r882's 1.13779 in 6 rounds; its median was 1.13443 and its best 1.13896;
- H217's GATED_025 mean, 1.13924, was above the live candidate in all 2,925 rounds;
- H210's GATED_025 mean, 1.13880, was above it in 2,921 of 2,922 rounds.

This is descriptive. The live loop trains a model each round; GATED_025 is a cross-fit selection rule averaged over
15 cells, not a single deployable checkpoint.

## Run record

- **Admission refusal (10:55Z, no claim).** xbabe2's disk was 93.86% used, over the 93.8% rule written for xbabe1,
  though about 477 GB was free. Amendment A1 (pre-outcome, before any claim) set xbabe2's stop to free < 200 GB and
  admission to free ≥ 220 GB. Record: `owner/admission-20261102T1055Z.json` in custody.
- **Attempt (`run-20261102T1115Z`, 11:15:43–13:09:20Z).** Admission passed at load 26.9/41.8 (bound 51), CPU 16%,
  472 GB free and 87 GiB available memory. The source check passed before the claim.
- **Guards.** They were sized from the 102-CPU effective affinity: stop at load 86.7 and worker tree > 4.5 cores. The
  absolute deadline was 15:15:43Z (claim + 4 h). There were 0 guard stops and no resume. The run finished PASS
  2 h 06 min before the deadline.
- **Resources.** Wall 6,814 s, fit wall 6,441 s, process CPU 20,418 s. Peak worker tree 3.92 cores, max RSS 3.7 GB,
  run root 155 MB.

## Caveats

- **Not a sealed test.** The scores are nested TRAIN cross-fit dev scores on the same 205 DL-Train families. Those
  families were already used by H124, H129, H195, H199, H203 and H210. No sealed, holdout or meta-test data was read.
- **Seeds are only partly independent.** Each seed has its own cross-fit family partition, distinct from H210's and
  H203's. All seeds and both waves share the 192-program training subsample per context and the deterministic r882
  TRAIN costs.
- **DL-Val is small.** 91 contexts in 19 families. The co-primary CI is a family bootstrap over those 19 families.
- **U4 does not confirm.** GATED_025's U4 CI crosses 0 in both H210 and H217.
- **Amendment A1.** It changed only xbabe2's disk rule, before any claim; science, seeds, folds, arms, decision
  rules and every other guard were unchanged.

## Verdict

**CONFIRMED: GATED_025 + NO_PLATEAU_WIN** on the flywheel native first-argmin protocol. It is a diagnostic result:
0 optimizer updates, DOPE frozen and no serving promotion.

## Shadow-only note

For `flywheel_gp_fused`, frozen DOPE `c51a56d0`, no serving change:
- **What.** GATED_025 would score each context with r882 and with BLEND_025. It takes the blend's pick only when
  `zgap ≤ τ`, and otherwise r882's pick. τ and the calibration come from FIT data only.
- **Status.** The gate is now confirmed on two independent partition sets, but it does not clear the plateau margin.
  The two read-only shadow logs continue; nothing in the live loop changes.
- **Not authorized.** Live use, a DOPE change or a serving change needs a separate GOV decision.

## Next step

- **Plateau.** The gated gain on DL-Val is about +0.001 to +0.0015, under the +0.002 margin. More of the ungated
  gain needs a better outcome-free signal or blend weight: H218 (rd_argon) tests a nested λ chosen jointly with the
  gate on FIT.
- **Harm.** Inspect the 9 harmed families per seed before any shadow decision.
- **Data.** A sealed or fresh-family check needs admitted JRB families; it was out of scope here.

## Custody

- **S3 (create-only, version-pinned readback):**
  `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H217/run-20261102T1115Z/`. It holds
  `run-outputs.tar.gz` (all 45 fits, panels, the lock, logs and the claim), `metrics.json`, `stage1_lock.json`,
  `execution-identity.json`, `launch.json`, the claim, the resource summary and `s3_manifest.json`.
- **Owner evidence.** The admission refusal is under `owner/`. The shadow helper, the logger and
  `gated_dlval-h217.json` are under `shadow/`.
- **Versions and SHA-256s.** Listed in [custody.json](custody.json).
- **Local.** The run root stays on xbabe2 at `~/jope-data/h217/run-20261102T1115Z`.
- **Receipts.** `~/jope_plan/ns/receipts/H217_claim.txt`.

**Code.** [runner.py](runner.py), [analysis.py](analysis.py), [monitor.py](monitor.py) and
[test_h217.py](test_h217.py) (34 data-free tests) are unchanged since amendment A1 merged.
