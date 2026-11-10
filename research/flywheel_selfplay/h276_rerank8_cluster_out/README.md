# H276: leave-family-cluster-out test of the confirmed RERANK8 re-ranker

**Status: complete. Verdict: PASS (research diagnostic; no serving or flywheel change).** The rule was preregistered in
[PREREG.md](PREREG.md) and published on `origin/v1/next` as `f709f8d0` at 20:32:21Z (PR #2071 split; same-agent review
receipt sha256 `7a25e4aa…`) before any outcome was read. The single attempt ran on xbabe2 CPU from 20:33:19Z to
21:12:42Z with 0 optimizer updates. The r882 sha256 and its DOPE identity `c51a56d0` were the same before and after.
Before compute, the smoke reproduced one H272 cell exactly using the byte-identical H272 science.

**RERANK8 holds up when whole family clusters are held out, and the H273 veto adds a little on top.**
- **PRIMARY, out-of-cluster on DL-Train** (4 outcome-blind partitions × 5 held-out clusters × 3 seeds = 60 cells):
  RERANK8 − r882 = **+0.0577 [+0.0367, +0.0738]**.
  - This uses a two-level cluster bootstrap. The Holm-level interval is [+0.0345, +0.0765], Holm p 0.0002.
  - 3/3 seeds positive (+0.0577, +0.0469, +0.0646), and **all 12 seed × partition values positive** (+0.025 to
    +0.085).
- **RERANK8_VETO** (H273 invalid gate, τ 0.10) − r882 = **+0.0686 [+0.0369, +0.0828]**, 3/3. VETO − RERANK8 = +0.0033
  [+0.0002, +0.0058], 3/3.
- **Invalid picks on DL-Train.** r882 204, RERANK8 186, **RERANK8_VETO 175**.
- **Placebo.** RERANK8_SHUF_UNC − r882 = −0.0050 [−0.0109, −0.0010], so the gate's ≤ +0.005 limit holds.
- **Clusters.** ≥ 4/5 clusters are non-negative in ≥ 2/3 seeds in every partition (KM276140 5, KM276141 5, KM276142 5,
  WARD 4).
- **Both gated arms pass all six gates.**

## Hypothesis

**What is known.** RERANK8 was replicated twice (H272 +0.0633; H273 clean-room +0.0571), but H272's leave-cluster-out
panel was weak: +0.0173 and not significant, below RERANK4, with more invalid picks.

**The question.** When whole outcome-blind family clusters are held out, does RERANK8 still lift single-pick utility
over r882 by more than +0.010? Does the H273 veto keep that lift without extra invalid picks?

## Setup

- **Frozen science.** [topk.py](topk.py) and [h265_core.py](h265_core.py) are byte-identical to H272/H270 (K = 8,
  features, labels, LGBMRanker grid, strict selector, margin rule).
  - **Regression:** H272 cell `F272140-S272101-K0` was reproduced exactly in both smokes (receipt, 2 invalid,
    131 changed, 84 deep picks).
  - **Checks:** Native91 was 91/91 bit-exact and TOP4_PICK@r882 matched exactly.
- **Arms.**
  - **CONTROL:** r882.
  - **RERANK4.**
  - **RERANK8.**
  - **RERANK8_VETO:** [veto.py](veto.py). The H273 `LGBMClassifier` predicts invalid candidates; it is fit on TRAIN
    rows only (inner-train folds for selection). It masks slots k ≥ 1 with p ≥ 0.10 before the unchanged margin
    selection. Its ranker streams equal RERANK8's.
  - **RERANK8_SHUF_UNC:** the placebo, with fallback disabled.
- **Clusters.** Built before outcomes from outcome-blind family descriptors (family-mean standardised context and pool
  features). They are k-means k = 5 with seeds 276140, 276141 and 276142, plus Ward k = 5, pinned in
  [fold_manifest.json](fold_manifest.json).
  - **Sizes:** KM276140 59/37/36/50/23; KM276141 73/49/59/23/1 (a singleton outlier family, kept as declared);
    KM276142 54/30/46/55/20; WARD 53/39/50/55/8.
  - **Inner selection:** family-grouped CV over 4 random folds of the training clusters only.
  - **Lock:** all 60 selections were fsync-locked (`bacc23ee…`) before any DL-Val91 read.
- **Statistics.** Per model seed, the gain is the median over the 4 partitions of the equal-family OOF gain.
  - **Bootstrap:** 10,000-draw two-level cluster bootstrap (seed 276150) that resamples clusters within each partition.
  - **Multiplicity:** Holm m = 2 over RERANK8 and RERANK8_VETO.
- **Resources.**
  - xbabe2 cores 44–47, nice 19, LightGBM 1 thread.
  - The monitor watches the swap-in rate, not free swap.
  - Peak RSS 0.90 GB, 3 threads; 2,374 s; no pause.

## Outcome

### PRIMARY: out-of-cluster DL-Train (1,219 contexts, 205 families; Elo with r882 = 1500)

| Contrast | Per-seed gain (median over partitions) | Median [CI95] | Holm-level CI | p | Seeds > 0 | W/T/L per seed | Elo |
|---|---|---|---|---:|---:|---|---:|
| **RERANK8 − r882** | +0.0577, +0.0469, +0.0646 | **+0.0577 [+0.0367, +0.0738]** | [+0.0345, +0.0765] | 0.0001 | 3/3 | 140/1/64, 136/5/64, 143/1/61 | 1556 |
| **RERANK8_VETO − r882** | +0.0692, +0.0417, +0.0686 | **+0.0686 [+0.0369, +0.0828]** | [+0.0343, +0.0864] | 0.0001 | 3/3 | 141/2/62, 136/6/63, 142/2/61 | 1551 |
| RERANK4 − r882 | +0.0178, +0.0258, +0.0247 | +0.0247 [+0.0113, +0.0328] | [+0.0097, +0.0339] | 0.0001 | 3/3 | 128/8/69, 130/4/71, 130/4/71 | 1526 |
| RERANK8 − RERANK4 | +0.0416, +0.0183, +0.0477 | +0.0416 [+0.0137, +0.0588] | — | 0.0003 | 3/3 | — | — |
| RERANK8_VETO − RERANK8 | +0.0043, +0.0013, +0.0033 | +0.0033 [+0.0002, +0.0058] | — | 0.014 | 3/3 | — | — |
| RERANK8_SHUF_UNC − r882 | −0.0050, −0.0027, −0.0079 | −0.0050 [−0.0109, −0.0010] | — | 0.997 | 0/3 | 48/104/53, 36/126/43, 60/61/84 | 1493 |

**Per seed × partition, RERANK8 − r882:**

| Seed | KM276140 | KM276141 | KM276142 | WARD |
|---|---|---|---|---|
| 276101 | +0.059 | +0.025 | +0.085 | +0.056 |
| 276102 | +0.042 | +0.053 | +0.036 | +0.052 |
| 276103 | +0.053 | +0.056 | +0.073 | +0.075 |

- **Worst cluster: WARD cluster 1** (39 families, 238 contexts).
  - RERANK8 − r882 by seed: −0.0005, −0.017 and +0.051 (median −0.0005).
  - Elo on that cluster: r882 1500, RERANK4 1530, **RERANK8 1563**, VETO 1562, placebo 1503.
  - The worst single seed × cluster value is −0.048 (KM276141 cluster 3, seed 276103).
  - The singleton cluster (KM276141 cluster 4) gains +0.078 to +0.100.
- **Admissions.** RERANK8 60/60, RERANK8_VETO 60/60, RERANK4 51/60, placebo 8/60 (strict; its fallback is disabled for
  scoring).

### SECONDARY: DL-Val91 (supporting only; heavily reused; the KM276140 cells' 5-model mixture per seed)

| Contrast | Per-seed gain | Median [CI95] | Seeds > 0 | Elo |
|---|---|---|---:|---:|
| RERANK8 − r882 | +0.0294, +0.0398, +0.0355 | +0.0355 [+0.0194, +0.0542] | 3/3 | 1582 |
| RERANK8_VETO − r882 | +0.0358, +0.0370, +0.0381 | +0.0370 [+0.0191, +0.0564] | 3/3 | 1599 |
| RERANK4 − r882 | +0.0273, +0.0145, +0.0225 | +0.0225 [+0.0078, +0.0402] | 3/3 | 1545 |
| RERANK8_SHUF_UNC − r882 | −0.0211, +0.0012, +0.0009 | +0.0009 [−0.0088, +0.0039] | 2/3 | 1499 |

- **Invalid picks on DL-Val91:** 0 for every arm except the placebo (2).
- **`fam-769243ee…`** (a DL-Val91 family) is again harmed by RERANK8 and RERANK8_VETO in **3/3** seeds.

## Reading

- **The lift generalises across held-out family clusters.** With each of 5 outcome-blind clusters held out in turn, 4
  partitions and 3 seeds, RERANK8 gains +0.058. Every one of the 12 seed × partition values is positive, and 4–5 of 5
  clusters are non-negative per partition. The H272 tertiary weakness (+0.017 on the 41 lm48-farthest families) does
  not recur at the cluster level here.
- **The veto helps modestly.** RERANK8_VETO adds about +0.003 over RERANK8 in 3/3 seeds and removes 11 invalid picks
  (175 against 186). Both are well below r882's 204.
- **Not every family wins.** About 61–64 of 205 families lose per seed, and one 39-family Ward cluster is flat
  (median −0.0005), though RERANK8 still wins more of its families (Elo 1563). The DL-Val91 family `fam-769243ee…`
  keeps losing across every wave.

## Self red-team

1. **The descriptor sets which families count as "dissimilar."** H276 clusters on a richer context-plus-pool
   descriptor with k = 5, so held-out clusters hold 23–73 families (one singleton). H272's tertiary held out the 41
   families farthest by `lm48_z` alone, and that tail is not identical to any H276 cluster. **Extreme outlier
   families remain the least-tested case.** The singleton and 8-family clusters are positive, but they are small.
2. **Five clusters per partition make a coarse bootstrap.** The CI is nevertheless clearly above 0 (Holm-level LCB
   +0.0345), and every seed × partition value is positive.
3. **The partitions share families.** The four partitions re-cut the same 205 families, so they are not independent
   replications. The seeds and partitions guard against cut-specific luck, not against a different population.
4. **Same code.** The same frozen code and feature set as H270/H272 are used. H273 (clean-room) is the
   independent-implementation check.
5. **DL-Val91 is heavily reused**, so it is supporting only. Its result agrees: +0.0355 and +0.0370, with 0 invalid
   picks.
6. **No sealed or holdout data was used.** This is the strongest held-out-of-selection check available without them.

## Verdict

**PASS** (RERANK8; RERANK8_VETO also passes). All six gates hold:
- median +0.0577 > +0.010;
- Holm-level LCB +0.0345 > 0;
- 3/3 seeds positive;
- ≥ 4/5 clusters non-negative in every partition;
- DL-Train −2.0 picks 186 ≤ 204;
- placebo −0.005 ≤ +0.005.

**This is not a serving promotion.** No checkpoint changed. Adoption is a GOV decision.

## Next step

1. **Bring the evidence to GOV:** H270 ADVANCE, H272 CONFIRMED, H273 clean-room REPLICATED, H276 cluster-out PASS.
   **RERANK8_VETO** is the preferred variant: a slightly larger lift and fewer invalid picks.
2. **If a sealed or holdout panel becomes available,** a single pre-registered RERANK8_VETO evaluation there would be
   the final check.
3. **Investigate `fam-769243ee…`** and the flat Ward cluster as abstention candidates.

## Files and custody

- **In this directory:**
  - [metrics.json](metrics.json): compact, 41 KB. It holds `dope_before`/`dope_after`, the decision and gates, the
    primary panel with per-partition and per-cluster gains, the DL-Val91 panel, the worst-cluster Elo, invalid counts,
    cluster sizes, admissions, per-cell summaries and the full-metrics pointer;
  - [PREREG.md](PREREG.md), [runner.py](runner.py), [veto.py](veto.py), [topk.py](topk.py), [h265_core.py](h265_core.py),
    [analysis.py](analysis.py), [analysis_h272.py](analysis_h272.py) (used only for the regression),
    [monitor.py](monitor.py) and [test_h276.py](test_h276.py) (8 data-free tests; registrations deferred under the
    freeze);
  - [manifest.json](manifest.json), [fold_manifest.json](fold_manifest.json) (17 KB clusterings) and
    [provenance.json](provenance.json).
- **S3, create-only with sha256 and VersionId readback:**
  - `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H276/attempt-001/` holds:
    - the full `metrics.json` (VersionId `TaiCogOi…`);
    - `selection.json`, including the partitions;
    - the resource summary and the attempt claim;
    - `custody-attempt-001.tar.gz`, which contains all 60 cell JSONs with picks, both panels, the stdout, the resource
      samples and both smokes;
    - `s3_manifest.json` (sha256 `0b878ff8…`, VersionId `kPp8pJgF…`).
  - The prereg sources, smokes and the review receipt are under `H276/prereg-f709f8d0/` (21 objects).
- **Not in Git.** No model or `.npz`.
