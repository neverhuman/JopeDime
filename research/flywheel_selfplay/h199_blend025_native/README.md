# H199 — BLEND_025 on the flywheel native first-argmin protocol, 205-family pooled cross-fit, 0 updates

**Status: COMPLETE. Verdict: NO_ADVANCE.** BLEND_025 raises mean native utility robustly (3/3 seeds, LCB95 > 0),
but it **fails the harm gate**: 24.9% of families are harmed, against a 10% cap. No adoption note is written,
because only a CANDIDATE gets one. No model, checkpoint or serving state changed.

## Hypothesis

H195 confirmed **BLEND_025 = r882 + 0.25·(GBM_cal − r882)** on H119 Track-2 U4. H135 (19 families) found no native
lift on flywheel91, but it was underpowered. H199 asks: **does BLEND_025 lift r882 on the flywheel native protocol**
(`fixed-dev-cost-argmin-negative-fitness/v1`: full-pool first argmin, −fitness when valid, −2.0 when invalid) **when
powered by the 205-family pooled cross-fit?** See [PREREG.md](PREREG.md).

## Setup

- **Prereg.** PR #1646, merged as `b4dd588e4` (ENG central squash).
  - The run used blobs equal to the accepted head `dfb6b8dc`. The reviewer's P2 host-guard block (review
    5730016603) was fixed there, and the final PR head `a32d8f77` only resolved merge conflicts.
- **Data.** 205 DL-Train families and 1,219 contexts, from real S3-derived data (the H124 pinned packet).
- **Seeds.** 199101–199103, with independent fold seeds 199141–199143. Best-match fold Jaccard is 0.16–0.17.
- **Cells and fits.** Cells are A = k, B = k+1, FIT = the other three. The H129 HistGBM and its positive-slope
  calibration come from FIT only: 45 BLEND fits plus 15 SHUF fits.
- **Lock.** The stage-1 lock (`c70993c0…`) was fsynced before any A or DL-Val scoring.
- **Primary.** Pooled-A native utility, checked on every context to equal H124's first-argmin utility. The bootstrap
  uses 10,000 paired family-cluster draws (seed 199150), reduced over seeds on axis 0.
- **Run.** One attempt on **xbabe1** CPU at nice 19:
  - 1 h 52 min wall;
  - peak 3.97 cores and 3.9 GB RSS;
  - root 171 MB;
  - **no resume**.
- **Frozen state.** DL-Val r882 native 91/91 was bit-exact first (1.1377941898164525). DOPE was `c51a56d0` before
  and after, with 0 updates.
- **Reads.** 0 sealed, holdout, meta-test or live-loop reads, and 0 DL-Val rows decoded by the TRAIN reader.

## Outcome: pooled A, 205 families, native first-argmin against R882 (Elo: r882 = 1500)

| Arm | Median gain | Per-seed | CI95 | Seeds > 0 | **Harm** | Elo [CI95] | −2.0 picks (R882: 17) |
|---|---:|---|---|---:|---:|---|---|
| **BLEND_025 (primary)** | **+0.0434** | +0.0429 / +0.0434 / +0.0437 | **[+0.0097, +0.0903]** | **3/3** | **24.9%** (51) | **1514.0** [1504.8, 1525.7] | 14 / 13 / 7 |
| BLEND_015 (dose) | +0.0484 | +0.0563 / +0.0268 / +0.0484 | [+0.0182, +0.0806] | 3/3 | 15.6% (32) | 1515.8 [1508.2, 1524.5] | 11 / 15 / 7 |
| BLEND_035 (dose) | +0.0491 | +0.0462 / +0.0533 / +0.0491 | [+0.0043, +0.1147] | 3/3 | 29.8% (61) | 1520.2 [1506.8, 1530.2] | 16 / 13 / 14 |
| BLEND_025_SHUF (1 seed) | +0.0014 | +0.0014 | [0.0000, +0.0037] | 1/1 | 0.0% | 1501.0 [1500.0, 1502.4] | 17 |

**Gates.**
- LCB95 > 0: ✓.
- Median > +0.001: ✓.
- 3/3 seeds: ✓.
- **Harm ≤ 10%: ✗ (24.9%).**

The verdict is **NO_ADVANCE**.

**U4, the H195 replication (report-only).** BLEND_025 gains +0.0118 [+0.0030, +0.0223], 3/3 seeds, harm 9.8%, Elo
1520.8. The dose gives +0.0059 at λ = 0.15 (harm 6.8%) and +0.0162 at λ = 0.35 (harm 11.2%). SHUF is null (Elo
1499.5). **This replicates H195 on fresh folds.**

**Flywheel91 DL-Val (91 contexts, 19 families, descriptive).** These are per-context mean native utilities against
R882's 1.13779:

| Arm | Mean native | vs R882 |
|---|---:|---:|
| BLEND_015 | 1.13506 | −0.0027 |
| BLEND_025 | 1.13552 | −0.0023 |
| BLEND_035 | 1.12602 | −0.0118 |
| SHUF | 1.13779 | identical |

No arm made a −2.0 pick. This agrees with H135 (median −0.0041).

## Family-grouped validation and interpretation

**Post-hoc family split** (descriptive, not preregistered; three-seed mean native delta per family):

| Arm | Families > +0.01 | Families < −0.01 | Families within ±0.01 | Seed-context native utilities changed (of 3,657) | Contexts with any changed utility (of 1,219) | Mean win | Mean loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| BLEND_015 | 63 | 32 | 110 | 747 (20.4%) | 404 (33.1%) | +0.174 | −0.065 |
| BLEND_025 | 71 | 51 | 83 | 1,155 (31.6%) | 584 (47.9%) | +0.195 | −0.099 |
| BLEND_035 | 80 | 61 | 64 | 1,505 (41.2%) | 717 (58.8%) | +0.223 | −0.127 |

The utility-change columns compare each arm's native utility with R882's, either per seed-context outcome (3 seeds ×
1,219 contexts) or per context in at least one seed. `panels.npz` records utilities, not selected candidate IDs. A
pick that moves to a candidate with the same utility is therefore not counted, and these columns are not
pick-change rates.

- **The native lift is real but not safe.** The blend changes the native utility in 31.6% of seed-context outcomes
  (47.9% of contexts in at least one seed). More families win than lose (71 against 51), and the wins are larger
  (+0.195 against −0.099), so the mean rises robustly. But one in four families is harmed by more than 0.01. U4 is
  buffered by the revealed slots and the top-4 set, which hides this redistribution (harm 9.8%).
- **Harm scales with λ.** The share of harmed families rises monotonically from 15.6% to 24.9% to 29.8%, while the
  mean gain stays flat at about +0.044 to +0.049. Shrinking λ lowers harm, but even at λ = 0.15 it does not reach
  the 10% cap.
- **The placebo is null** (Elo 1501). The movement comes from what the GBM learned, not from blending mechanics.
- **Flywheel91 is consistent with this.** Its 19 families are too few to show a mean gain against a quarter of
  families being harmed, which explains H135's null.

## Verdict

**NO_ADVANCE.** BLEND_025 (and the λ = 0.15 and 0.35 variants) must **not** be adopted into the flywheel's
first-argmin selection. It is positive on average but harms about 25% of families.

## Next step

Any further work should target the harm directly, with no re-run of this blend. A preregistered **family-safe
gate** is one option: apply the blend only where an outcome-free, FIT-calibrated signal predicts no loss (for
example GBM/r882 agreement on the argmin neighbourhood). The success criterion should be harm ≤ 10% at a native
gain with LCB95 > 0. This is H129's gated idea (BLEND_050_GATED) re-asked on the native endpoint. Any wave stays
shadow-only with DOPE frozen.

## Custody

- **S3 (create-only, version-pinned readback):**
  `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H199/run-20261101T2034Z/`. It holds 7
  objects (including `run-outputs.tar.gz`, with all 60 fits, panels, locks and logs) and `s3_manifest.json`. Version
  IDs and SHA-256s are in [custody.json](custody.json).
- **Local.** The run root stays on xbabe1 at `~/jope-data/h199/run-20261101T2034Z` (171 MB).
- **Receipts.** `~/jope_plan/ns/receipts/H199_claim.txt`.

**Code.** [runner.py](runner.py), [analysis.py](analysis.py), [monitor.py](monitor.py) and
[test_h199.py](test_h199.py) (20 data-free tests) are unchanged since the merged prereg.
