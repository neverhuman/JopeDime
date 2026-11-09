# H267: bagged median-rank top-4 re-ranker (zero-update, single-pick)

**Status: complete. Verdict: NO_ADVANCE.** The rule was preregistered in [PREREG.md](PREREG.md) and published on
`origin/v1/next` as `d5760f8b` at 18:11:29Z (PR #2071 split; same-agent review receipt sha256 `3de3d826…`) before any
outcome was read. The single attempt ran on xbabe2 CPU from 18:12:07Z to 18:22:30Z, with 0 optimizer updates. The r882
sha256 and its DOPE identity `c51a56d0` were the same before and after.

**Bagging did not stabilise the re-ranker; it suppressed it.**
- **The decision.** On the PRIMARY DL-Train out-of-fold panel (205 families), BAG_MED − r882 = 0.0000
  [−0.0030, +0.0091] (Holm-level [−0.0049, +0.0116]). Per seed it is 0, −0.0163 and +0.0110: 1/3 positive, Holm p 0.87.
  It does not beat its placebo, and SECONDARY DL-Val91 is positive in only 2/3 seeds (+0.0041, Holm p 0.051). All five
  gates fail.
- **The bagged arm was rarely admitted.** Its out-of-bag (OOB) gain over r882 was positive in **15/15** cells (+0.006
  to +0.060), while the placebo's was negative in 13/15. But the OOB LCB95 cleared 0 in only **3/15** cells, so BAG_MED
  fell back to r882 in 12/15. Where it was admitted, it switched picks more often than RERANK4 in 2 of the 3 cells (183 vs
  150 and 121 vs 75; 175 vs 179 in the third), and seed 267102's two admitted cells lost out of fold.
- **The H265 single model replicated on DL-Val91 with fresh partitions (descriptive).** RERANK4 − r882 = +0.0161
  [+0.0050, +0.0289] on DL-Val91, 3/3 seeds (+0.0146, +0.0173, +0.0161), so the per-seed spread is small. On TRAIN out
  of fold it is +0.0174 [−0.0093, +0.0580], 3/3 seeds but p 0.11, as in H265, with 49–67 harmed families per seed.

## Hypothesis

**What is known.** H265 RERANK4 advanced on DL-Val91 (+0.0187), with three red-team caveats: seed instability, 3–5
harmed families per seed, and a non-significant 205-family DL-Train OOF result (p 0.062).

**The question.** Does family-bootstrap bagging (B = 25) with median-rank aggregation make the re-ranker's lift stable
and positive on DL-Train out of fold, while keeping it on DL-Val91?

## Setup

- **Frozen from H265.** [h265_core.py](h265_core.py) is a byte-identical copy of the `v1/next` H265 runner (sha256
  pinned), so the features, labels, LGBMRanker recipe and grid, strict selector and margin rule were used unchanged. The
  H265 tests (5) and the H267 tests (10) passed before compute.
- **Arms.**
  - **CONTROL:** r882 slot 0.
  - **RERANK4:** H265 per cell.
  - **BAG_MED:** 25 LGBMRankers on family bootstraps of the 164 FIT families, at the cell's H265-selected
    (lr, n_est). It picks the slot with the smallest median within-context rank, ties to r882.
  - **BAG_SHUF:** the same with labels shuffled within family.
  - **Admission of the bagged arms:** out-of-bag gain > .001 and LCB95 > 0, else r882.
- **Design.**
  - Repeated nested family CV with one fresh fold seed per model seed (267101↔267140, 267102↔267141,
    267103↔267142): 15 cells, three independent partitions.
  - DL-Val91 is scored with the same cells. All selections were fsync-locked (`selection.json` sha256 `9ed5270f…`)
    before any DL-Val91 read.
  - **Runtime note.** The pairing was frozen on a 190 s/cell estimate (3.0 s per synthetic fit). The realised cell
    time was about 40 s (623 s in total), so the full 45-cell cross would have fit before the stop. The paired design
    was kept as preregistered.
- **Checks.** Native91 1.1377941898164525 was 91/91 bit-exact, and TOP4_PICK@r882 reproduced (context
  1.2403601682250796, family 1.2519997168684853).
- **Resources.** xbabe2 cores 44–47, nice 19, LightGBM 1 thread; peak RSS 0.90 GB, 3 threads; no pause.

## Outcome

### PRIMARY: DL-Train out of fold (1,219 contexts, 205 families; Elo with r882 = 1500)

| Contrast | Per-seed gain | Median [CI95] | Holm-level CI | p | Seeds > 0 | W/T/L per seed | Elo |
|---|---|---|---|---:|---:|---|---|
| **BAG_MED − r882** | 0, −0.0163, +0.0110 | **0.0000 [−0.0030, +0.0091]** | [−0.0049, +0.0116] | 0.87 | 1/3 | 0/205/0, 44/124/37, 26/165/14 | 1507 [1500, 1514] |
| RERANK4 − r882 | +0.0379, +0.0078, +0.0174 | +0.0174 [−0.0093, +0.0580] | [−0.0134, +0.0643] | 0.11 | 3/3 | 80/76/49, 93/56/56, 116/22/67 | 1526 [1513, 1538] |
| BAG_MED − RERANK4 | −0.0379, −0.0241, −0.0064 | −0.0241 [−0.0524, −0.0014] | — | — | 0/3 | — | — |
| BAG_MED − BAG_SHUF | 0, −0.0163, +0.0110 | 0.0000 [−0.0030, +0.0091] | — | — | 1/3 | — | — |
| BAG_SHUF − r882 | 0, 0, 0 | 0.0000 [0, 0] | — | — | 0/3 | 0/205/0 | 1500 |

### SECONDARY: DL-Val91 single-pick (91 contexts, 19 families; each seed's 5-cell mixture)

| Contrast | Per-seed gain | Median [CI95] | Holm-level CI | Seeds > 0 | Harmed families per seed | Elo [CI95] |
|---|---|---|---|---:|---|---|
| **BAG_MED − r882** | 0, +0.0081, +0.0041 | **+0.0041 [0, +0.0093]** | [0, +0.0102] | 2/3 | 0, 4, 7 | 1544 [1500, 1588] |
| RERANK4 − r882 | +0.0146, +0.0173, +0.0161 | +0.0161 [+0.0050, +0.0289] | [+0.0037, +0.0310] | 3/3 | 4, 4, 6 | 1548 [1504, 1607] |
| BAG_MED − RERANK4 | −0.0146, −0.0092, −0.0119 | −0.0119 [−0.0225, −0.0034] | — | 0/3 | — | — |
| BAG_SHUF − r882 | 0, 0, 0 | 0.0000 [0, 0] | — | 0/3 | 0, 0, 0 | 1500 |

- **Holm m = 2** over {PRIMARY, SECONDARY} for BAG_MED − r882: adjusted p 0.87 and 0.051.
- **Admissions.** RERANK4 13/15 cells, BAG_MED 3/15 (S267102-F1, S267102-F2, S267103-F4), BAG_SHUF 0/15.
- **Invalid (−2.0) picks.** On DL-Val91, 0 for every arm. On TRAIN out of fold, RERANK4 and BAG_MED are within ±4 of
  r882 per cell.
- **Recurring harmed families.** The same DL-Val91 families recur across arms and seeds (`fam-280b1a13…`,
  `fam-538e6406…`, `fam-469b72b7…`, `fam-769243ee…`, `fam-7d4e2dc9…`).
- **Context #27** (`fam-9f67f144…`): 1.1227 in every arm and seed (unchanged).

## Reading

- **Bagging carries real signal, but the strict OOB rule throws most of it away.** Every cell's OOB gain was positive
  and the placebo's was mostly negative. However, each FIT context is OOB in only about 9 of the 25 bags, so the
  per-family OOB estimate is noisy, and the LCB95 > 0 gate rejected 12/15 cells.
- **Median-rank aggregation removes RERANK4's margin protection.** H265's margin rule switches away from r882's slot 0
  only on a clear score gap. The median-rank rule switches whenever another slot's median rank is strictly lower. In 2 of
  the 3 admitted cells it changed more choices than RERANK4, and in seed 267102 that lost out of fold.
- **The H265 single model is the stronger result here.** With three fresh, independent partitions it lifts DL-Val91 by
  +0.016 in 3/3 seeds with a tight per-seed spread. That eases H265's seed-instability caveat on DL-Val91, but its
  205-family TRAIN out-of-fold lift is still not significant (p 0.11) and it harms 49–67 TRAIN families per seed.

## Verdict

**NO_ADVANCE.**
- BAG_MED − r882 on PRIMARY DL-Train OOF: 0.0000 [−0.0030, +0.0091], 1/3 seeds, Holm p 0.87.
- It does not beat BAG_SHUF.
- SECONDARY is positive on only 2/3 seeds.
- No serving or live-flywheel change. r882 and DOPE are unchanged.

## Next step

**Do not pursue bagged median-rank aggregation as specified.**

**If ensembling is revisited**, aggregate bag **scores** (mean), not ranks, and keep H265's margin rule on the
aggregate. Admit on a pooled inner CV rather than per-family OOB.

**The more informative follow-up is RERANK4's TRAIN out-of-fold harm.** It harms 49–67 TRAIN families per seed, and
the same DL-Val91 families recur across arms and seeds. A harm-aware margin, or abstaining on those family types, is the
natural next hypothesis. H266 (replication and ablations) covers the plain-replication question.

## Files and custody

- **In this directory:**
  - [metrics.json](metrics.json): the compact readout, 31 KB, with `dope_before`/`dope_after`, the decision and gates,
    both panels, per-cell admissions and OOB gains, invalid and changed counts, the cliff context and the full-metrics
    pointer. Per-context picks are in S3, not Git;
  - [PREREG.md](PREREG.md), [runner.py](runner.py), [analysis.py](analysis.py), [h265_core.py](h265_core.py),
    [monitor.py](monitor.py), [test_h267.py](test_h267.py) (10 data-free tests; registrations deferred under the
    freeze), [manifest.json](manifest.json), [fold_manifest.json](fold_manifest.json) and
    [provenance.json](provenance.json).
- **S3, create-only with sha256 and VersionId readback:**
  - `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H267/attempt-001/` holds:
    - the full `metrics.json` (VersionId `IlUSLCdg…`);
    - `selection.json`;
    - the resource summary and the attempt claim;
    - `custody-attempt-001.tar.gz`, which contains every cell JSON, `val_detail.json` with the per-context picks, the
      panels, the stdout, the resource samples and both smokes;
    - `s3_manifest.json` (sha256 `9a106a0f…`, VersionId `REfd_V6x…`).
  - The prereg sources, smokes and the review receipt are under `H267/prereg-d5760f8b/` (18 objects).
- **Not in Git.** No model or `.npz`. The bags are recomputable from the pinned inputs and seeds.
