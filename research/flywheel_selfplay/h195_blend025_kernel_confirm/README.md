# H195 — prospective confirmation of H129 BLEND_025 on fresh folds (kernel-active r882), 0 updates

**Status: COMPLETE. Verdict: CANDIDATE: BLEND_025.** All three preregistered gates pass. This is a **candidate for
the `flywheel_gp_fused` owner to consider; it is not adopted**, and no model, checkpoint or serving state changed.

## Hypothesis

H129 found **BLEND_025 = r882 + 0.25·(GBM_cal − r882)** (cost space, calibrated HistGBM prior) a CANDIDATE. Its
three seeds shared one fold partition, and its prospective confirmation, H134, never reached a readout. H195
re-asks the question prospectively, on **three independent family partitions**: does BLEND_025 beat r882 on U4 on
families the GBM never saw?

**Premise correction (registered).** H129's r882 was already kernel-active, because the H124 baseline reveals slots
0..3 to the live adapter. The primary is therefore the exact H129 specification. The KERNEL-removed variants are
descriptive. See [PREREG.md](PREREG.md).

## Setup

- **Prereg.** PR #1607, merged as `10ba7d575`. The run used blobs equal to the reviewed head `08aa6641`.
- **Review.** rd_argon did not review within the 15-minute cap, so the cross-review was waived on the PR. The
  waiver, the merger's verification review 5729412444 and the same-agent review are bound into `launch.json`.
- **Seeds.** 195101–195103, with independent fold seeds 195141–195143. Best-match fold Jaccard is 0.15–0.16, which is
  chance level.
- **Cells.** A = k, B = k+1, FIT = the other three.
- **GBM.** The H129 HistGBM recipe, with 192 sampled programs per context. The GBM and its positive-slope
  calibration come from FIT only:
  - BLEND: 45 fits;
  - SHUF placebo: 15 fits, seed 195101 only.
- **Lock.** The stage-1 lock (`4fc6ba33…`) was fsynced before any A or DL-Val scoring.
- **Primary.** BLEND_025 against R882 on pooled-A U4, Holm m = 1. Bootstrap: 10,000 paired family draws, seed
  195150, reduced over seeds on axis 0.
- **Run.** One attempt on **xbabe1** CPU at nice 19:
  - 1 h 44 min wall;
  - peak 3.95 cores and 3.8 GB RSS;
  - root 170 MB;
  - **no resume**.
- **Frozen state.** DL-Val r882 native 91/91 was bit-exact first. The H184 decomposition was bit-exact on every
  context. DOPE was `c51a56d0` before and after, with 0 updates.
- **Reads.** 0 sealed, holdout, meta-test or live-loop reads, and 0 DL-Val rows decoded by the TRAIN reader.

## Outcome (pooled A, 205 families, against R882)

| Arm | U4 median gain | Per-seed | CI95 | p vs +0.001 | Harm | Elo vs r882 [CI95] | −2.0 native picks |
|---|---:|---|---|---:|---:|---|---|
| **BLEND_025 (primary)** | **+0.0137** | +0.0137 / +0.0241 / +0.0097 | **[+0.0056, +0.0259]** | **0.0005** | **7.3%** (15) | **1523.1** [1513.5, 1532.4] | 10 / 18 / 13 (R882: 17) |
| BLEND_025_NOKERNEL | +0.0140 | +0.0133 / +0.0278 / +0.0140 | [+0.0050, +0.0289] | 0.0010 | 10.7% (22) | 1512.7 [1500.5, 1524.8] | 19 / 28 / 17 |
| BLEND_025_SHUF (1 seed) | +0.000005 | +0.000005 | [−0.00005, +0.00008] | 1.0 | 0.0% | 1499.3 [1497.5, 1500.7] | 18 |
| R882_NOKERNEL | −0.0032 | −0.0032 (seed-invariant) | [−0.0150, +0.0098] | 0.75 | 12.7% (26) | 1477.5 [1466.1, 1489.3] | 20 |

**Gates.** LCB95 > 0, p < 0.05 and harm ≤ 10% all pass: **CANDIDATE: BLEND_025**.

**Native argmin (descriptive).** BLEND_025 gains +0.042 [+0.013, +0.084] over R882 on the flywheel-style first-argmin
pick. R882_NOKERNEL loses −0.052 [−0.083, −0.018].

**DL-Val (91 contexts, descriptive).** This readout uses the mean calibrated outer-model prior and equal-family
means. Arm values are listed in [metrics.json](metrics.json).
- BLEND_025 native: 1.1612 against R882's 1.1536.
- U4: 1.2798 against 1.2777.
- 0 −2.0 picks.

## Metrics and interpretation

- **The H129 candidate replicates prospectively.** On three fresh, independent partitions the gain is +0.0137 (H129:
  CI [+0.0065, +0.0273]). Elo is 1523.1 (H129: 1523.5). All three seeds are positive, and the native argmin also
  improves, with fewer invalid picks in 2 of 3 seeds.
- **The placebo is null.** The SHUF GBM, trained on within-family permuted targets, early-stops at 10–12 iterations
  (BLEND runs 300). Its calibration hits the 0.05 slope floor in 3 of 5 cells, and its blend changes nothing
  (+0.000005, Elo 1499). The gain therefore comes from what the GBM learns about the program–context interaction, not
  from blending mechanics.
- **The kernel still matters for r882.** Removing it costs r882 −0.0032 U4 and −0.052 native, with Elo 1477.5.
  - The GBM blend recovers most of this on U4 (NOKERNEL blend +0.0140 against R882). It does so with more harm
    (10.7%, just over the cap), lower Elo (1512.7) and more −2.0 picks.
  - So the GBM prior and the kernel are partly substitutable on U4. The kernel-active blend is the safer combination.
- **Caveats.**
  - U4 is H119 Track-2, not the flywheel's own DL-Val protocol (H135's question), and DL-Val here is descriptive.
  - SHUF is one seed.
  - λ = 0.25 was not searched. That is the point of the confirmation, but it also means no dose information.
  - p is the H129 bootstrap tail against the +0.001 margin, at its resolution floor (1/10,001 per draw).

## Verdict

**CANDIDATE: BLEND_025.** It is prospectively confirmed on U4 against r882, at 0 updates with DOPE frozen. It is
**not adopted** and is flagged to the `flywheel_gp_fused` owner.

## Next step

The owner decides whether a frozen-DOPE flywheel shadow should score a BLEND_025 prior. The prior would come from a
FIT-only GBM, with no λ search, on the flywheel's own DL-Val protocol (H135's question), using a fresh held-out
partition, before any serving change. A λ dose curve, if wanted, should be its own preregistered wave.

## Custody

- **S3 (create-only, version-pinned readback):**
  `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H195/run-20261101T1800Z/`. It holds 7
  objects (`run-outputs.tar.gz`, containing all 60 fits, panels, locks and logs, plus key JSONs) and
  `s3_manifest.json`. Version IDs and SHA-256s are in [custody.json](custody.json).
- **Local.** The run root stays on xbabe1 at `~/jope-data/h195/run-20261101T1800Z` (170 MB).
- **Receipts.** The claim and launch receipts are at `~/jope_plan/ns/receipts/H195_claim.txt`.

**Code.** [runner.py](runner.py), [analysis.py](analysis.py), [monitor.py](monitor.py) and
[test_h195.py](test_h195.py) (11 data-free tests) are unchanged since the prereg.
