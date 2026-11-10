# H272: fresh-split confirmation of the H270 RERANK8 re-ranker, with depth arms

**Status: complete. Verdict: CONFIRMED (research confirmation; no serving or flywheel change).** The rule was
preregistered in [PREREG.md](PREREG.md) and published on `origin/v1/next` as `f629cb79` at 19:14:23Z (PR #2071 split;
same-agent review receipt sha256 `842fd669…`) before any outcome was read. The single attempt ran on xbabe2 CPU from
19:15:21Z to 20:05:58Z with 0 optimizer updates. The r882 sha256 and its DOPE identity `c51a56d0` were the same before
and after. Before compute, the smoke reproduced one real H270 cell exactly (receipt and all 247 picks), using the
byte-identical H270 source.

**RERANK8 replicates on fresh family splits, with a slightly larger lift than H270.**
- **PRIMARY, DL-Train out of fold** (205 families, full 45-cell cross on never-used fold seeds): RERANK8 − r882 =
  **+0.0633 [+0.0232, +0.1250]**.
  - Holm-level interval [+0.0189, +0.1351], Holm p 0.0002.
  - 3/3 seeds positive (+0.0666, +0.0603, +0.0633), and **all 9 seed × fold-seed repetitions positive** (+0.035 to
    +0.078).
  - H270 had +0.0523 [+0.0171, +0.1060].
- **Placebo and baselines.**
  - RERANK8_SHUF − r882 = 0 (placebo admitted 3/45; RERANK8 admitted 45/45).
  - RERANK8 − RERANK4 = +0.0484 [+0.0171, +0.0807], 3/3.
- **SECONDARY, DL-Val91:** RERANK8 − r882 = +0.0335 [+0.0193, +0.0509], 3/3 seeds, Holm p 0.0002. DL-Val91 is heavily
  reused.
- **Invalid picks.** On DL-Train, RERANK8 makes **137** −2.0 picks against r882's 153. On DL-Val91 it makes 1 across
  the 15 cells, against 0 (H270: 2).
- All six gates pass.

**Two descriptive caveats limit how far this generalises:**
- **On the 41 least-similar families** (the TERTIARY leave-cluster-out panel), RERANK8 − r882 shrinks to +0.0173
  [−0.0314, +0.0836] (3/3 positive, not significant). There RERANK8 is *below* RERANK4 (−0.0125) and adds invalid picks
  (12 against r882's 9 over 3 seeds).
- **Deeper slices pay on TRAIN but not on DL-Val91.**
  - RERANK12 and RERANK16 beat r882 by +0.087 and +0.083 out of fold, and beat RERANK8 by +0.020 and +0.029.
  - On DL-Val91, however, both are *below* RERANK8 (−0.016 and −0.015), with **19 and 11 invalid picks** against
    RERANK8's 1.

## Hypothesis

**What is known.** H270 ADVANCED (RERANK8 +0.0523 on DL-Train OOF), but in one run, with a reused DL-Val91, 2 DL-Val91
invalid picks and 64–68 losing families per seed.

**The question.** Does RERANK8's lift replicate on fresh, never-used family splits and seeds, using byte-identical
code? And, descriptively, how does the lift change at depths 12 and 16?

## Setup

- **Frozen sources.** [topk.py](topk.py) and [h265_core.py](h265_core.py) are byte-identical to the H270 execution
  source (`fb33cb6a1` = `aee5dd042`), pinned and tested. Only the seeds change for RERANK8; the depth arms change
  only K.
- **Arms.**
  - **CONTROL:** r882.
  - **RERANK8:** confirmatory.
  - **RERANK8_SHUF:** placebo.
  - **RERANK4:** reference.
  - **RERANK12 and RERANK16:** descriptive, not gated.
- **Design.**
  - **PRIMARY:** a full 45-cell cross on fresh fold seeds 272140/141/142 × model seeds 272101–272103. Per seed, the
    gain is the median over fold seeds. Family bootstrap 10,000 draws, boot seed 272150.
  - **Depth arms:** the full 45-cell cross as well, run after the confirmatory lock. Not truncated.
  - **SECONDARY:** DL-Val91, from the fold-seed-272140 cells.
  - **TERTIARY:** the 41 families farthest from the mean family centroid of standardised TRAIN `lm48_z` descriptors
    (outcome-free, pinned before outcomes) are held out. The other 164 families train the unchanged selector.
  - **Locks:** A (confirm, `7d5b8757…`), B (depth, `f644d49d…`) and C (tertiary, `2379f6e6…`) were all fsync-locked
    before any DL-Val91 read.
- **Checks.**
  - Native91 91/91 bit-exact and TOP4_PICK@r882 exact.
  - The test files for H265 (5), H270 (10) and H272 (8) passed before compute.
- **Resources.** xbabe2 cores 44–47, nice 19; peak RSS 0.95 GB, 3 threads; 3,033 s; no pause.

## Outcome

### PRIMARY: DL-Train out of fold (1,219 contexts, 205 families; Elo with r882 = 1500)

| Contrast | Per-seed gain (median over fold seeds) | Median [CI95] | Holm-level CI | p | Seeds > 0 | W/T/L per seed | Elo [CI95] |
|---|---|---|---|---:|---:|---|---|
| **RERANK8 − r882** | +0.0666, +0.0603, +0.0633 | **+0.0633 [+0.0232, +0.1250]** | [+0.0189, +0.1351] | 0.0002 | 3/3 | 147/5/53, 148/2/55, 144/3/58 | 1559 [1537, 1575] |
| RERANK4 − r882 | +0.0244, +0.0162, +0.0126 | +0.0162 [−0.0065, +0.0475] | [−0.0094, +0.0523] | 0.08 | 3/3 | 116/12/77, 123/26/56, 121/7/77 | 1525 [1509, 1538] |
| RERANK8 − RERANK4 | +0.0484, +0.0543, +0.0394 | +0.0484 [+0.0171, +0.0807] | [+0.0141, +0.0878] | 0.0005 | 3/3 | — | — |
| RERANK8 − RERANK8_SHUF | +0.0748, +0.0603, +0.0633 | +0.0633 [+0.0237, +0.1278] | — | 0.0002 | 3/3 | — | — |
| RERANK8_SHUF − r882 | 0, 0, 0 | 0.0000 [0, 0] | — | 1.0 | 0/3 | 1/196/8, 0/205/0, 0/205/0 | 1500 |

- **Per (seed × fold seed)**, RERANK8 − r882 is:
  - seed 272101: +0.078, +0.061, +0.067;
  - seed 272102: +0.071, +0.035, +0.060;
  - seed 272103: +0.052, +0.071, +0.063.
- **Family utility.** r882 1.3854 → RERANK8 1.4405–1.4537.
- **Oracles.** The top-8 oracle is 1.6360 and the top-4 oracle 1.5787.

### SECONDARY: DL-Val91 (91 contexts, 19 families)

| Contrast | Per-seed gain | Median [CI95] | Holm-level CI | Seeds > 0 | Harmed families per seed | Elo [CI95] |
|---|---|---|---|---:|---|---|
| **RERANK8 − r882** | +0.0279, +0.0335, +0.0347 | **+0.0335 [+0.0193, +0.0509]** | [+0.0179, +0.0536] | 3/3 | 2, 2, 1 | 1609 [1555, 1667] |
| RERANK4 − r882 | +0.0187, +0.0103, +0.0217 | +0.0187 [+0.0019, +0.0331] | [−0.0009, +0.0351] | 3/3 | 4, 3, 3 | 1579 [1525, 1633] |
| RERANK8 − RERANK4 | +0.0092, +0.0232, +0.0130 | +0.0130 [+0.0011, +0.0284] | — | 3/3 | — | — |

**Recurring harmed family.** `fam-769243ee…` loses under RERANK8 in all three seeds here, as it did in H265, H267 and
H270. `fam-538e6406…` loses in one seed.

### TERTIARY: leave-family-cluster-out (41 least-similar families, 201 contexts; descriptive)

| Contrast | Per-seed gain | Median [CI95] | Seeds > 0 | W/T/L per seed | Invalid picks per seed (r882: 3, 3, 3) |
|---|---|---|---:|---|---|
| RERANK8 − r882 | +0.0038, +0.0536, +0.0173 | +0.0173 [−0.0314, +0.0836] | 3/3 | 22/6/13, 25/5/11, 29/0/12 | 6, 2, 4 |
| RERANK4 − r882 | +0.0168, +0.0410, +0.0298 | +0.0298 [−0.0222, +0.0848] | 3/3 | 19/6/16, 15/21/5, 23/4/14 | 4, 2, 3 |
| RERANK8 − RERANK4 | −0.0130, +0.0126, −0.0125 | −0.0125 [−0.0580, +0.0435] | 1/3 | — | — |

### Depth curve (descriptive, full 45-cell cross; all depth arms admitted in 45/45 cells)

| Readout | RERANK8 | RERANK12 | RERANK16 |
|---|---|---|---|
| DL-Train OOF − r882 [CI95] | +0.0633 [+0.023, +0.125] | +0.0874 [+0.036, +0.155] | +0.0833 [+0.046, +0.131] |
| DL-Train OOF − RERANK8 | — | +0.0202 [−0.004, +0.046], 3/3 | +0.0293 [+0.0002, +0.046], 3/3 |
| DL-Train −2.0 picks (r882: 153) | 137 | 116 | 105 |
| DL-Train harmed families per seed | 53, 55, 58 | 62, 49, 57 | 46, 40, 52 |
| DL-Val91 − r882 [CI95] | +0.0335 [+0.019, +0.051] | +0.0187 [−0.047, +0.074] | +0.0156 [−0.027, +0.055] |
| DL-Val91 − RERANK8 | — | −0.0160, 0/3 | −0.0154, 0/3 |
| DL-Val91 −2.0 picks (r882: 0) | 1 | **19** | **11** |

- **Context #27** (`fam-9f67f144…`): RERANK8 1.1250–1.1262 against r882 1.1227 (no flip).
- **Invalid picks across arms.** On DL-Train: r882 153, RERANK4 151, RERANK8 137, SHUF 154, RERANK12 116, RERANK16 105.

## Reading

- **The H270 result is real on fresh splits.** With the same frozen code and new fold and model seeds, RERANK8 gains
  +0.063 out of fold. All nine repetitions are positive, the placebo stays at 0, and it is tighter on DL-Val91 than in
  H270. It also makes fewer invalid TRAIN picks than r882.
- **It transfers less well to dissimilar families.** On the 41 families least like the rest, the lift shrinks to
  +0.017 and is not significant. There RERANK8 is no better than RERANK4 and adds a few invalid picks. The re-ranker
  learns patterns that hold within the bulk of the family distribution.
- **Deeper is not safely better.** K = 12 and 16 keep improving TRAIN out of fold, but on DL-Val91 they fall below
  K = 8 and make 11–19 invalid picks where r882 makes none. A plausible reason, not tested here, is that DL-Val91 contexts
  hold gate-failing candidates in slots 8–15 that the ranker cannot recognise. K = 8 is the best-supported depth.

## Verdict

**CONFIRMED.** All six preregistered gates pass:
- PRIMARY +0.0633 > +0.002;
- Holm-level LCB +0.0189 > 0;
- 3/3 seeds positive;
- beats RERANK8_SHUF;
- SECONDARY 3/3 positive;
- DL-Train invalid picks 137 ≤ 153.

**This is not a serving promotion.** No checkpoint changed (0 updates; r882 and DOPE identical). Adopting RERANK8 into
the flywheel is a separate GOV decision. The cluster-holdout weakness and the planned independent-lane code
re-implementation should inform that decision.

## Next step

1. **An independent-lane re-implementation** of RERANK8 (already queued by the watcher).
2. **Harm-aware gating.** An invalid-pick veto or abstain rule for `fam-769243ee…`-like contexts and for out-of-cluster
   families, so the gain holds on dissimilar families.
3. **Keep K = 8.** If deeper slices are revisited, add a validity-aware feature or veto before going past 8.

## Files and custody

- **In this directory:**
  - [metrics.json](metrics.json): the compact readout, 42 KB, with `dope_before`/`dope_after`, the decision and gates,
    the primary, secondary, tertiary and depth panels, invalid counts, oracles, admissions and per-cell summaries, the
    cliff context and the full-metrics pointer. Per-context picks are in S3, not Git;
  - [PREREG.md](PREREG.md), [runner.py](runner.py), [topk.py](topk.py), [h265_core.py](h265_core.py),
    [analysis.py](analysis.py), [monitor.py](monitor.py) and [test_h272.py](test_h272.py) (8 data-free tests;
    registrations deferred under the freeze);
  - [manifest.json](manifest.json), [fold_manifest.json](fold_manifest.json) (47 KB, including the held-out cluster)
    and [provenance.json](provenance.json).
- **S3, create-only with sha256 and VersionId readback:**
  - `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H272/attempt-001/` holds:
    - the full `metrics.json` (VersionId `N3epm.hT…`);
    - the three selection locks;
    - the resource summary and the attempt claim;
    - `custody-attempt-001.tar.gz`, which contains all cell JSONs (confirm and depth), the panels, the stdout, the
      resource samples and both smokes;
    - `s3_manifest.json` (sha256 `bd30ecea…`, VersionId `sdeH0xiA…`).
  - The prereg sources, smokes and the review receipt are under `H272/prereg-f629cb79/` (19 objects).
- **Not in Git.** No model or `.npz`.
