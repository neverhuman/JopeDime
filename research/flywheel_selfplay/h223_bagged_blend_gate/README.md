# H223 — bagged blend component inside the frozen GATED_025 recipe, xbabe2 CPU, 0 updates

**Status: COMPLETE. Verdict: NO_CANDIDATE + NO_PLATEAU_WIN.** Averaging three HistGBM members before the 0.25 blend
did not help. The bagged gate's native gain over r882 has a CI that crosses 0, with 2/3 seeds positive. The bagged
blend's harm is not lower than the single blend's. The DL-Val plateau gain is +0.0005, far below the +0.002 margin.
The frozen GATED_025 reference, run alongside, replicated its H210/H217 result a third time. No model, checkpoint,
DOPE or serving state changed.

## Hypothesis

[H217](../h217_gated025_confirm/README.md) confirmed GATED_025 (native +0.0216, harm 4.4%) but missed the DL-Val
plateau margin (+0.00145 < +0.002). Its ungated BLEND_025 gained +0.0367 with 25.9% harm. **H223 (hw_t1325) asks:**
if the blend's harm comes from a high-variance single HistGBM per context, does averaging three members lower that
harm, let the FIT-only `zgap` τ open more often, and lift DL-Val past +0.002? See [PREREG.md](PREREG.md).

## Setup

- **Prereg.** PR #1865, merged as `530eef65` (a merger rebased the reviewed head `60bf707c`; the H223 folder was
  identical). Amendment A1 (PR #1873, `fcda9bd3`, hw_t1400): host load pauses the worker instead of aborting.
  Amendment A2 (PR #1884, `be474a54`, hw_t1500): one continuation after the 17:25Z deadline stop, with a fixed 21:30Z
  compute deadline. Both amendments were pre-outcome and changed no science.
- **Bag.** Each of the 45 fit units fits three HistGBM members on one design matrix (135 fits). Member 0 is H217's
  unweighted fit, tested bit-identical, and doubles as GATED_025's component. Members 1–2 are family-cluster
  bootstrap refits, using family multiplicities as `sample_weight`. The BAG prior is the mean of the three members,
  with its own FIT-only positive-slope calibration.
- **Gate.** H210/H217's `zgap` gate and FIT-only τ rule, frozen: maximize FIT native gain subject to FIT harm ≤ 5%,
  else r882. Function bodies are tested identical to H217.
- **Data.** 205 DL-Train families and 1,219 contexts from the H124 pinned packet (real S3-derived data, staging hash
  list `2f7a3642…`).
- **Seeds.** 223101–223103, with fold seeds 223141–223143. Best-match fold Jaccard is at most 0.194 against H203,
  H210, H217 and H218.
- **Frozen state.** DL-Val r882 native 91/91 bit-exact in every attempt (1.1377941898164525). DOPE `c51a56d0` before
  and after, 0 optimizer updates, no GPU.
- **Reads.** 0 sealed, holdout, meta-test or live-loop reads.

## Outcome: pooled A, 205 families, native first-argmin against R882 (Elo: r882 = 1500)

| Arm | Median gain | Per-seed | CI95 | p (vs +0.001) | Seeds > 0 | Harm | Elo [CI95] | −2.0 picks (R882: 17/17/17) |
|---|---:|---|---|---:|---:|---:|---|---|
| **GATED_BAG3 (primary)** | **+0.0012** | +0.0012 / −0.0109 / +0.0247 | **[−0.0015, +0.0048]** | 0.49 | **2/3** | 5.4% (11) | 1503.1 [1497.1, 1507.9] | 17 / 19 / 12 |
| GATED_025 (frozen reference) | +0.0080 | +0.0080 / +0.0170 / +0.0062 | [+0.0012, +0.0190] | 0.022 | 3/3 | 7.3% (15) | 1504.3 [1498.6, 1507.5] | 16 / 13 / 14 |
| GATED_BAG3_DIS (placebo) | −0.0006 | −0.0015 / +0.0001 / −0.0006 | [−0.0020, +0.0003] | 1.00 | 1/3 | 2.0% (4) | 1500.9 [1499.3, 1502.7] | 17 / 17 / 17 |
| BLEND_BAG3 (ungated bag) | +0.0406 | +0.0408 / +0.0245 / +0.0406 | [+0.0100, +0.0689] | 0.003 | 3/3 | 28.8% (59) | 1517.4 [1503.4, 1527.1] | 10 / 14 / 14 |
| BLEND_025 (ungated single) | +0.0396 | +0.0492 / +0.0251 / +0.0396 | [+0.0086, +0.0773] | 0.005 | 3/3 | 26.8% (55) | 1513.7 [1504.0, 1524.8] | 15 / 14 / 15 |

**Primary gates for GATED_BAG3.**
- LCB95 > 0: ✗ (−0.0015).
- 3/3 seeds positive: ✗ (seed 223102 is −0.0109).
- Harm ≤ 10%: ✓ (5.4%).

The primary is **NO_CANDIDATE**.

**Paired GATED_BAG3 − GATED_025 (report-only).** Native −0.0068 [−0.0196, +0.0014], Elo 1498.9 [1495.5, 1503.3].
DL-Val −0.0003 [−0.0008, +0.00004]. The bagged gate is no better than the single-model gate, and on its point
estimates slightly worse.

**Co-primary: flywheel91 DL-Val (91 contexts, 19 families; r882 1.13779).**

| Arm | Context mean | Gain | CI95 |
|---|---:|---:|---|
| **GATED_BAG3** | **1.13829** | **+0.00050** | **[−0.00012, +0.00139]** |
| GATED_025 | 1.13861 | +0.00082 | [+0.00015, +0.00189] |
| GATED_BAG3_DIS | 1.13823 | +0.00043 | [−0.00093, +0.00174] |
| BLEND_025 | 1.13660 | −0.00120 | [−0.02258, +0.01498] |
| BLEND_BAG3 | 1.13333 | −0.00447 | [−0.02851, +0.01511] |

The gain is below +0.002 and the LCB is below 0, so the co-primary is **NO_PLATEAU_WIN**. No arm made a −2.0 pick on
DL-Val.

**U4 (report-only).** GATED_BAG3 +0.0139 [+0.0026, +0.0281], 3/3 seeds, Elo 1513.8. GATED_025 +0.0103
[+0.0022, +0.0244], Elo 1512.0. The bagged gate does better on U4 than on native, but U4 is not a decision endpoint.

## Gate behaviour

- **Switching.** The bagged blend's pick differs from r882's in 33–37% of A contexts per seed; the single blend's in
  31–38%. Bagging did not make the blend switch less often.
- **Open rates among switching contexts.** GATED_BAG3 13.0% / 18.5% / 20.0%; GATED_025 17.0% / 18.4% / 14.6%. The
  placebo opens on 2.9–6.5%.
- **τ per cell (FIT only).** GATED_BAG3's τ is 0.0013–0.0145 in 14 cells, and its gate stayed closed in `223102/4`.
  GATED_025's gate stayed closed in `223103/1`. FIT harm is ≤ 4.9% in every cell.
- **Interpretation.**
  - The variance-reduction premise did not hold here. The ungated bagged blend harms *more* families than the single
    blend (59 vs 55). Its gain is similar (+0.0406 vs +0.0396).
  - The FIT-only gate did not open more often on the bag.
  - On A, the bagged gate's selection was noisier across seeds than the single-model gate: −0.0109 on seed 223102.
  - Member 0 is shared with the reference, so the bag is "full fit + 2 family-bootstrap refits". This is a
    conservative test of variance reduction (see PREREG design decision). Three independent bootstraps might behave
    differently, but nothing here suggests a large effect.

## Run record

All three executions share one run root, `~/jope-data/h223/run-20261102T1405Z` on xbabe2 (CPU, nice 19, OMP 7).

1. **Attempt (`fcda9bd3`), 14:05:28–17:25:07Z.** Stopped at its preregistered 17:25Z deadline, with 19 of 45 units
   done. It was paused for load (A1) for 5,337 s over 12 events. Peak RSS 3.71 GB, peak tree 6.96 cores. An ETA
   warning was posted at 14:17Z, and the owner then authorized A2.
2. **Continuation (`be474a54`), 18:00:10Z–19:27Z.** It reused the 19 hash-verified units. Admission waited from
   17:26Z to 18:00Z because of a load spike from other jobs. At 19:27:00Z xbabe2 was **rebooted** (Jepson-approved
   GPU0 Xid79 recovery), killing the worker with 34 of 45 units done. The in-flight unit left no files.
3. **Post-reboot resume (`be474a54`, hw_t1945).** All 102 member files hash-verified against their receipts. The
   first continuation's claim, identity files and regression/train directories were preserved by rename
   (`*.interrupted-reboot-1927Z`), so the identical merged code could run one more resume. It reused the 34 verified
   units, kept the same guards and the 21:30Z compute stop, was admitted at 19:50:18Z, and finished PASS at 20:28Z
   (2,255 s, 0 load pauses, peak RSS 3.57 GB, peak tree 6.91 cores). After the reboot the host's effective CPU
   affinity was 128, not 102. This changes only the sustained load1/CPU pause threshold (0.85 × cores); the core,
   RSS, memory and disk caps are absolute.

- **Controller.** The continuation controller's start condition was tightened at 16:01Z (handback
  check-20261102T1553Z). It only continues after the original 17:25Z deadline stop. The receipt is
  `~/jope_plan/evidence/H223/continuation-guard-v2-receipt.json`.
- **Note on main's code.** After A2, jankurai dedupe PRs #1932 and #1936 refactored H223's `analysis.py` and
  `monitor.py` on main. The run executed the pinned `be474a54` code, not main's. On main, one H223 monitor test now
  fails, because the A1 pause became an abort; this was flagged to that lane.

## Caveats

- **Not a sealed test.** The scores are nested TRAIN cross-fit dev scores on the same 205 DL-Train families used by
  H124–H217. No sealed, holdout or meta-test data was read.
- **Partly independent seeds.** The seeds have distinct fold partitions but share the 192-program subsample and the
  r882 costs.
- **The bag is not three independent bootstraps.** Member 0 is the full fit; see the design decision in the PREREG.
- **Three executions.** The run spanned an attempt, a continuation and a post-reboot resume. Every reused unit was
  hash-verified, all three used the same seeds, folds and members, and the stage-1 lock (`aef8af6f…`) was written
  once, after all 45 units.
- **DL-Val is small.** 91 contexts in 19 families.

## Verdict

**NO_CANDIDATE + NO_PLATEAU_WIN** on the flywheel native first-argmin protocol: reject GATED_BAG3. Under hw_t1530's
rule (ADVANCE only if the plateau gain is > +0.002 with LCB > 0), this is **NO_ADVANCE**. It is a diagnostic: 0
optimizer updates, DOPE frozen, no serving promotion.

## Next step

- **Do not pursue member bagging of the HistGBM blend component.** Gain did not improve and harm did not fall.
- **GATED_025 is now replicated on three partition sets** (H210, H217, H223). Its DL-Val gain stays at about
  +0.0008 to +0.0015, under the plateau margin. A larger lift needs a different outcome-free signal or a better base
  model, not a lower-variance blend.
- **Fix main's H223 monitor.** The jankurai dedupe (#1932/#1936) turned the A1 pause into an abort; flagged to
  that lane.

## Custody

- **S3 (create-only, version-pinned readback):**
  `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H223/run-20261102T1405Z/`:
  - `attempt/`: the first attempt's archive, unit-hash manifest, claim, launch file, summary and identity;
  - `final/`: the final run root and logs archive, both claims, the interruption evidence and the controller scripts;
  - `s3_manifest.json`.
- **Prereg.** `.../H223/prereg-60bf707c/`.
- **Versions and SHA-256s.** Listed in [custody.json](custody.json).
- **Receipts.** `~/jope_plan/ns/receipts/H223_claim.txt`; evidence is in `~/jope_plan/evidence/H223/`.

**Code.** [runner.py](runner.py), [analysis.py](analysis.py), [monitor.py](monitor.py) and
[test_h223.py](test_h223.py). The run executed `be474a54` (A2); main's copies were later refactored by #1932/#1936.
