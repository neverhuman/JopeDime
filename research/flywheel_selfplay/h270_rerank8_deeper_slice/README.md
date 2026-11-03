# H270: re-ranking a deeper slice (top-8) of r882's own pool (zero-update, single-pick)

**Status: complete. Verdict: ADVANCE (research diagnostic; no serving or flywheel change).** The rule was preregistered
in [PREREG.md](PREREG.md) and published on `origin/v1/next` as `fb33cb6a` at 18:36:01Z (PR #2071 split; same-agent
review receipt sha256 `9acd4a38…`) before any outcome was read. The single attempt ran on xbabe2 CPU from 18:36:39Z
to 18:59:49Z with 0 optimizer updates. The r882 sha256 and its DOPE identity `c51a56d0` were the same before and after.

**Re-ranking r882's top-8 instead of its top-4 lifts out-of-fold utility about 5× more (+0.0523 against +0.0099).**
- **PRIMARY, DL-Train out of fold** (205 families, full 45-cell cross on fresh folds): RERANK8 − r882 = **+0.0523
  [+0.0171, +0.1060]**.
  - Holm-level interval [+0.0125, +0.1156], Holm p 0.0018.
  - 3/3 model seeds positive (+0.0489, +0.0523, +0.0593), and **all 9 (model seed × fold seed) repetitions positive**
    (+0.037 to +0.079).
- **Placebo and baselines.**
  - RERANK8_SHUF − r882 = 0. The placebo was admitted in only 2/45 cells; RERANK8 was admitted in 45/45.
  - RERANK8 − RERANK4 = +0.0390 [+0.0133, +0.0886], 3/3.
- **SECONDARY, DL-Val91:** RERANK8 − r882 = +0.0346 [+0.0111, +0.0604], 3/3 seeds, Holm p 0.0018. DL-Val91 is heavily
  reused, so DL-Train OOF on fresh folds is the decision endpoint.
- **Invalid picks.** RERANK8's DL-Train −2.0 picks (150 over all 45 ASSESS folds) are not above r882's (153). On
  DL-Val91 RERANK8 makes 2 invalid picks across the 15 cells, against 0 for r882 (reported, not gated).
- All six preregistered gates pass.

## Hypothesis

**What is known.** H265–H267 showed that the r882 top-4 re-ranker captures only about +0.01–0.02 of the top-4 oracle
headroom, and H266 showed the whole-pool oracle far above the top-4 oracle. Candidate generation looked like the
bottleneck, and H227/H229 showed that wider *planner* pools do not help.

**The question.** Does re-ranking a deeper slice of r882's **own** pool (its top-8), with the same zero-update ranker,
lift DL-Train out-of-fold single-pick utility over r882?

## Setup

- **Only K changes.**
  - [topk.py](topk.py) is the K-generic form of the frozen H265 pipeline ([h265_core.py](h265_core.py), a
    byte-identical copy). At **K = 4 it reproduces H265 exactly**: tests assert equal features, labels, LightGBM
    predictions, run_cell receipts and picks.
  - At K = 8: z-features are computed within the 8-slice, pairwise differences cover the other 7 slots (width 568), and
    labels run 7…0 with gate-failed candidates worst.
  - The LGBMRanker groups and truncation are 8. The grid, the strict inner selector and the margin rule are unchanged.
- **Arms.**
  - **CONTROL:** r882 top-1.
  - **RERANK4:** H265 frozen, run with H265's random streams.
  - **RERANK8 (primary):** slots 0–7.
  - **RERANK8_SHUF (placebo):** labels permuted within family.
- **Design.**
  - Full 45-cell cross: fold seeds 270140/141/142 × 5 folds × model seeds 270101–270103. FIT is 164 families and
    ASSESS is 41. Per model seed, the gain is the median over fold seeds of the equal-family OOF gain. Family bootstrap
    10,000 draws, boot seed 270150.
  - DL-Val91 is scored with the fold-seed-270140 cells.
  - All 45 selections were fsync-locked (`selection.json` sha256 `e3498f3c…`) before any DL-Val91 read.
- **Checks.**
  - Native91 1.1377941898164525 was 91/91 bit-exact, and TOP4_PICK@r882 reproduced exactly. The H265 and H270 test
    files passed before compute.
  - **Every slice is full:** pools hold 1,151–1,152 candidates.
- **Leakage audit (added at closeout).**
  - The features use only r882 costs, the base prior cost, the program, the sem24/proj16 embeddings and context
    descriptors, never fitness or gate.
  - The inherited "rows 0–3" descriptor is not outcome-ordered: those rows sit at the 0.49 gated-cost percentile, and
    hold the oracle best in 2/1,219 contexts against 4.2 expected at random.
- **Resources.** xbabe2 cores 44–47, nice 19, LightGBM 1 thread; peak RSS 0.84 GB, 3 threads; 1,391 s; no pause.

## Outcome

### PRIMARY: DL-Train out of fold (1,219 contexts, 205 families; Elo with r882 = 1500)

| Contrast | Per-seed gain (median over fold seeds) | Median [CI95] | Holm-level CI | p | Seeds > 0 | Family W/T/L per seed | Elo [CI95] |
|---|---|---|---|---:|---:|---|---|
| **RERANK8 − r882** | +0.0489, +0.0523, +0.0593 | **+0.0523 [+0.0171, +0.1060]** | [+0.0125, +0.1156] | 0.0009 | 3/3 | 136/1/68, 138/3/64, 138/3/64 | 1562 [1543, 1583] |
| RERANK4 − r882 | +0.0099, +0.0202, +0.0058 | +0.0099 [−0.0087, +0.0340] | [−0.0115, +0.0378] | 0.13 | 3/3 | 111/24/70, 121/14/70, 119/13/73 | 1521 [1507, 1534] |
| RERANK8 − RERANK4 | +0.0390, +0.0322, +0.0570 | +0.0390 [+0.0133, +0.0886] | [+0.0095, +0.0958] | 0.002 | 3/3 | 133/0/72, 133/3/69, 139/1/65 | — |
| RERANK8 − RERANK8_SHUF | +0.0489, +0.0523, +0.0610 | +0.0523 [+0.0174, +0.1066] | — | 0.0009 | 3/3 | — | — |
| RERANK8_SHUF − r882 | 0, 0, 0 | 0.0000 [0, 0] | — | 1.0 | 0/3 | 0/205/0 (×2), 0/200/5 | 1500 |

- **Per (model seed × fold seed)**, RERANK8 − r882 is:
  - seed 270101: +0.0755, +0.0410, +0.0489;
  - seed 270102: +0.0523, +0.0544, +0.0454;
  - seed 270103: +0.0785, +0.0368, +0.0593.
- **Family utility.** r882 1.3854 → RERANK8 1.4361–1.4436.
- **Headroom.** The top-8 oracle is 1.6360 and the top-4 oracle 1.5787, so RERANK8 recovers about 21% of the r882 →
  top-8 oracle gap (+0.251).
- **Picks.** 37.5% of RERANK8's out-of-fold picks come from slots 4–7. It changed 6,246 of 10,971 assessed picks;
  RERANK4 changed 3,065.

### SECONDARY: DL-Val91 (91 contexts, 19 families; each seed's 5-cell mixture at fold seed 270140)

| Contrast | Per-seed gain | Median [CI95] | Holm-level CI | Seeds > 0 | Harmed families per seed | Elo [CI95] |
|---|---|---|---|---:|---|---|
| **RERANK8 − r882** | +0.0346, +0.0386, +0.0290 | **+0.0346 [+0.0111, +0.0604]** | [+0.0086, +0.0651] | 3/3 | 2, 2, 6 | 1588 [1540, 1643] |
| RERANK4 − r882 | +0.0079, +0.0140, +0.0127 | +0.0127 [+0.0034, +0.0229] | [+0.0023, +0.0245] | 3/3 | 3, 2, 4 | 1570 [1527, 1608] |
| RERANK8 − RERANK4 | +0.0267, +0.0245, +0.0163 | +0.0245 [+0.0025, +0.0458] | [−0.0003, +0.0496] | 3/3 | — | — |
| RERANK8_SHUF − r882 | 0, −0.0011, 0 | 0.0000 [0, 0] | — | 0/3 | 0, 1, 0 | 1500 |

- **Holm m = 2** over {PRIMARY, SECONDARY} for RERANK8 − r882: adjusted p 0.0018 and 0.0018.
- **Invalid (−2.0) picks.** On DL-Train ASSESS (45 cells): r882 153, RERANK4 163, **RERANK8 150**, RERANK8_SHUF 153.
  On DL-Val91 (15 cells): RERANK8 2, all others 0. 231 of the 9,752 TRAIN top-8 slots are gate-failed, against 11 of
  728 on DL-Val91.
- **Recurring DL-Val91 harmed families.** `fam-769243ee…` (all 3 seeds), `fam-280b1a13…` and `fam-d73e184e…`.
- **Context #27** (`fam-9f67f144…`): RERANK8 1.1250–1.1273 against r882 1.1227. Small gains, no flip.
- **Selected settings.** RERANK8's margin was 0 in 8 cells, 0.02–0.1 in 21 and 0.2–0.5 in 16. Its inner FIT gain was
  +0.054 to +0.108 (mean +0.080).

## Reading

- **The useful candidates were already in r882's pool, one rank band deeper.** Widening the re-ranked slice from 4 to
  8 adds the oracle-best candidate often enough that the same ranker, trained only to order the slice, gains about
  5× more than at K = 4 out of fold (+0.0523 against +0.0099). That held on three fresh partitions and three model seeds, with every repetition
  positive. It also lifts DL-Val91 by about 2.7× more.
- **This is a real ranking signal, not a selection artefact.** The label-shuffled placebo is rejected by the inner
  selector in 43/45 cells and is 0 on both panels. RERANK8 is admitted in 45/45 cells with large inner gains. It does
  not add invalid picks on DL-Train (150 against 153).
- **It is not free of harm.** About 64–68 of the 205 TRAIN families lose per seed (against 136–138 that win), and
  DL-Val91 has 2–6 harmed families per seed. Two of those recur from H265/H267. RERANK8 makes 2 invalid picks on
  DL-Val91 where r882 makes none.
- **Candidate generation is partly a ranking problem inside the existing pool.** The planner pools need not get wider
  (H227/H229). r882's own top-8 already holds most of the reachable gain at this ranker capacity.

## Verdict

**ADVANCE**, a research-diagnostic advance on the preregistered rule. All six gates pass:
- PRIMARY median +0.0523 > +0.002;
- Holm-level LCB +0.0125 > 0;
- 3/3 seeds positive;
- beats RERANK8_SHUF (+0.0523);
- SECONDARY 3/3 positive (+0.029 to +0.039);
- DL-Train invalid picks 150 ≤ 153.

**This is not a serving promotion.** No checkpoint changed (0 updates; r882 and DOPE identical). Deploying a re-ranker
is a separate decision (GOV), and it requires an independent replication and a check off the reused DL-Val91.

## Next step

1. **Independent replication** of RERANK8 by another lane on fresh fold seeds, ideally with a held-out panel that is
   not DL-Val91.
2. **Map the depth curve.** Test K ∈ {12, 16, 32} with the same frozen pipeline, to find where deeper slices stop
   paying (more invalids and noisier labels).
3. **Make it harm-aware.** Add an abstain or margin rule targeting the recurring harmed families and DL-Val91 invalid
   picks, so the lift holds without the losses.

## Files and custody

- **In this directory:**
  - [metrics.json](metrics.json): the compact readout, 36 KB, with `dope_before`/`dope_after`, the decision and gates,
    both panels with per-repetition gains, invalid counts, oracle headroom, per-cell admissions, margins and
    deep-pick counts, the leakage audit, the cliff context and the full-metrics pointer. Per-context picks are in S3,
    not Git;
  - [PREREG.md](PREREG.md), [runner.py](runner.py), [topk.py](topk.py), [h265_core.py](h265_core.py),
    [analysis.py](analysis.py), [monitor.py](monitor.py), [test_h270.py](test_h270.py) (10 data-free tests;
    registrations deferred under the freeze), [manifest.json](manifest.json), [fold_manifest.json](fold_manifest.json)
    and [provenance.json](provenance.json).
- **S3, create-only with sha256 and VersionId readback:**
  - `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H270/attempt-001/` holds:
    - the full `metrics.json` (VersionId `Fr9Rj4n7…`);
    - `selection.json`;
    - the resource summary and the attempt claim;
    - `custody-attempt-001.tar.gz`, which contains all 45 cell JSONs with ASSESS picks, `val_detail.json`, both
      panels, the stdout, the resource samples and both smokes;
    - `s3_manifest.json` (sha256 `8a9ed6a3…`, VersionId `Nnakz6v2…`).
  - The prereg sources, smokes and the review receipt are under `H270/prereg-fb33cb6a/` (19 objects).
- **Not in Git.** No model or `.npz`. All models are recomputable from the pinned inputs and seeds.
