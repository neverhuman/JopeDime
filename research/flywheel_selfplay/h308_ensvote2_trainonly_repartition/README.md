# H308: H303 NESTED ENS_VOTE2 through a TRAIN-only loader, on two new outer partitions

**Verdict: `PARTITION_ROBUST`.** This is shadow research only. It does not authorize serving, and fresh confirmation
still needs a new S3 ingest (H299/H300).

**What was tested.** H303's frozen nested ENS_VOTE2 was re-fitted on two new outer family partitions, P1 and P2, with a
clean-room loader that decodes no DL-Val row.

**Result.**
- Both partitions beat r882 slot0: median family-equal gain **+0.0482** (P1) and **+0.0436** (P2).
- The per-seed LCB95 is > 0 in **6/6** (partition, seed) cells, and the drop-top-5 LCB95 is > 0 in **6/6** cells.
- **The gain is not partition luck.** H303's original partition P0 (+0.0643) sits on the high side, though: on fresh
  partitions the effect is about 0.045.

**Elo: reference = r882 slot0 (DL-Train OOF); r3175-anchored Elo unmeasured.**

**Records.**
- **Prereg:** [PREREG.md](PREREG.md), commit `2e2a22eda5d67c2ac65e591b76e343d533c53088`, pushed non-force to
  `origin/v1/next` at 07:35:23Z. The push included the loader, tests, partition salts and decision rule. It came before
  any new-partition fit (07:42:33Z) or score (08:16:50Z).
- **Owner:** rd_opus [OPUS].
- **Assignment:** `~/nudges/hw_t0735_opus_h308.md`.

## Hypothesis

H305 left two holes in H303's NESTED_CONFIRMED:
- **(h) Loader.** The loaders decode the 91 DL-Val feature rows of the mixed cache NPZs, and the "0 outer-test rows read
  before lock" counters overstate isolation.
- **One partition.** All ENS_VOTE2 evidence uses one outer partition, P0, whose seeds 30002/30003 are near-duplicates
  (family-gain corr .98).

H308 asks: with a loader that never touches DL-Val, and on partitions never used before, does the frozen H303 procedure
still beat r882 slot0?

## Setup

### Step 0: TRAIN-only loader ([trainonly_loader.py](trainonly_loader.py))

**What it reads.**
- Only the `cids` and `offsets` identifiers.
- It verifies that the 1,219 TRAIN cids are exactly the leading prefix of the 1,310 rows (pool rows `[0, 1404287)` of
  1,509,119).
- It `pread`s only that prefix byte range of each uncompressed feature member.

**Counters for the whole run.**
- 2,811,012 feature rows read, which is exactly 2 × 1,404,287 + 2 × 1,219.
- 98.8 MB read.
- **0 DL-Val feature rows.**

**Negative fixture ([test_h308.py](test_h308.py), 7 tests).** The trailing rows of a fixture NPZ are poisoned in place
with NaN:
- `np.load` fails its CRC on them;
- the loader returns the exact clean prefix and never reads them;
- a non-prefix layout is refused.

**Honest counters.** All TRAIN packets are built once, before the fork, so each task's outer-test packets are present in
memory before its lock (7,314 = 1,219 × 6 in total). A guarded packets dict raises on any outer-test access inside a
task until its fit_lock exists. Result: **0 accesses and 0 guard denials in 24/24 tasks** (`guard_receipt.json` per
task, in S3).

### Step 1: regression first, with the new loader: PASS ([regression_receipt.json](regression_receipt.json))

H303's P0 fold 0 / seed 30001 reproduced the archived H303 lock exactly:
- the choice (0.05, 5, 2);
- inner gain 0.0961251;
- the 18-row inner table;
- every member model hash;
- the fold-0 picks, sha `18fdb2538cea…`.

### Step 2: REPARTITION

H303 NESTED was frozen and imported with sha256 pins. Everything below is unchanged:
- H287 `213b30ad52` and `selected.json` `981960ad…`;
- the 18-config grid and inner 4-fold family CV;
- the 1e-4 conservative ties;
- seeds 30001–30003 and their member seeds.

The new outer partitions:
- **P1** = SHA256("H308-outer-v1" NUL family) mod 4: 53/50/50/52 outer-test families per fold.
- **P2** = SHA256("H308-outer-v2" NUL family) mod 4: 48/56/59/42 per fold.

**Run.**
- 24 tasks (936 member fits) on xbabe2 cores 100–102, 112, 113, 115, 123, 125, nice 19, 8 workers, `MemoryMax=8G`
  (never hit; the watchdog never fired).
- Fit 07:42:33–08:16:41Z.
- The 24 lock hashes were inventoried, then each outer_test was scored once.

**P0** = H303's numbers, recomputed from its archived picks. There was no refit, and the recomputation matched H303 exactly.

## Outcome (out-of-fold, 1,219 contexts / 205 families, vs r882 slot0)

| partition | seed | family-equal gain [95% CI] | LCB95 | family median | W/T/L | harmed | invalid vs 17 (excess, McNemar p) | switches | Elo [95% CI] |
|---|---|---|---|---|---|---|---|---|---|
| **P1** | 30001 | +0.04821 [+0.01802, +0.09301] | +0.02163 | +0.0030 | 112/57/36 | 11.2% | 17 (0, 0.623) | 327 | 1697.1 [1636.3, 1768.5] |
| **P1** | 30002 | +0.04866 [+0.02219, +0.08470] | +0.02523 | +0.0036 | 116/51/38 | 10.2% | 15 (−2, 0.891) | 275 | 1693.8 [1633.8, 1761.9] |
| **P1** | 30003 | +0.03829 [+0.01601, +0.06006] | +0.01963 | +0.0033 | 112/64/29 | 8.3% | 16 (−1, 0.746) | 285 | 1734.7 [1668.1, 1812.7] |
| **P1** | **median** | **+0.04821** [pooled +0.02219, +0.08393] | pooled +0.02487 | +0.0033 [0, +0.0077] | | max 11.2% | max excess 0 | | **1697.1** [1651.7, 1767.1] |
| **P2** | 30001 | +0.04360 [+0.01768, +0.07196] | +0.02148 | +0.0063 | 122/48/35 | 10.7% | 16 (−1, 0.726) | 315 | 1716.9 [1656.1, 1787.7] |
| **P2** | 30002 | +0.03184 [+0.01027, +0.05343] | +0.01393 | +0.0036 | 112/65/28 | 7.8% | 17 (0, 0.688) | 292 | 1740.8 [1674.8, 1820.1] |
| **P2** | 30003 | +0.05472 [+0.03384, +0.08018] | +0.03660 | +0.0081 | 129/41/35 | 10.7% | 15 (−2, 0.891) | 335 | 1726.6 [1666.0, 1796.1] |
| **P2** | **median** | **+0.04360** [pooled +0.01880, +0.07055] | pooled +0.02258 | +0.0063 [+0.0016, +0.0098] | | max 10.7% | max excess 0 | | **1726.6** [1675.3, 1790.2] |
| P0 (H303) | median | +0.06432 [pooled +0.02229, +0.13359] | pooled +0.02771 | +0.0016 | | max 9.3% | max excess −3 | | 1725.7 [1667.0, 1790.2] |

Per-seed P0 values (H303): gains +0.0413/+0.0643/+0.0723, LCB95 +0.0245/+0.0209/+0.0287.

### Concentration

| partition | seed | gain after dropping top 1 / 5 / 10 | top-10 share | drop-top-5 LCB95 |
|---|---|---|---|---|
| P1 | 30001 | +0.0307 / +0.0183 / +0.0102 | 79.9% | **+0.0086** |
| P1 | 30002 | +0.0353 / +0.0233 / +0.0141 | 72.5% | **+0.0120** |
| P1 | 30003 | +0.0333 / +0.0210 / +0.0127 | 68.4% | **+0.0064** |
| P2 | 30001 | +0.0373 / +0.0212 / +0.0107 | 76.7% | **+0.0052** |
| P2 | 30002 | +0.0265 / +0.0149 / +0.0088 | 73.8% | **+0.0011** |
| P2 | 30003 | +0.0486 / +0.0340 / +0.0236 | 59.0% | **+0.0226** |
| P0 | 30001 / 30002 / 30003 | (H303) | 69.7 / 87.2 / 78.1% | +0.0124 / −0.0003 / +0.0071 |

### Chosen configs and seed dependence

| partition | TAU 0.05 / 0.10 / 0.15 | k 5 / 6 / 7 | a = 2 | H303 design point (0.10, 6, 2) | family-gain corr (01–02 / 01–03 / 02–03) |
|---|---|---|---|---|---|
| P0 | 8 / 2 / 2 | 8 / 3 / 1 | 12/12 | 1 | .27 / .31 / **.98** |
| P1 | 4 / 5 / 3 | 9 / 1 / 2 | 12/12 | 0 | **.93** / .28 / .42 |
| P2 | 7 / 2 / 3 | 7 / 4 / 1 | 12/12 | 0 | .77 / .75 / .64 |

**Observations.**
- The near-duplicate seed pair is **partition-specific** (P0: 30002/30003; P1: 30001/30002; P2: none above .77), so it
  is not a property of the seeds.
- Agreement a = 2 is chosen in 36/36 tasks.
- k = 5 is the modal choice on every partition.

## Verdict

**Pre-registered rule.**
1. **PASS:** each new partition's median is > +0.001 with pooled LCB95 > 0 (P1 +0.0482 / +0.0249; P2 +0.0436 / +0.0226).
2. **PASS:** the per-seed LCB95 is > 0 in 6/6 cells (needed 5).
3. **PASS:** the drop-top-5 LCB95 is > 0 in 6/6 cells (needed 4).
4. **PASS:** harmed is ≤ 11.2% in every cell.
5. **PASS:** invalid excess is ≤ 0 in every cell.

**PARTITION_ROBUST.**

**Reading.**
- The H303 gain survives a clean TRAIN-only loader and two fresh partitions, with a stronger concentration profile than
  P0: drop-top-5 is positive in 6/6 new cells against 2/3 on P0, and the family median is positive in every new cell.
- The size is smaller off P0: medians +0.044–0.048 against P0's +0.064. P0 was a favourable split, so **about +0.045 is
  the better central estimate** of ENS_VOTE2 NESTED vs r882 slot0 on DL-Train.
- Harmed is somewhat higher than P0 (7.8–11.2% against 6.3–9.3%).

## Deviations

None from the prereg. P2 was not dropped; all 24 tasks ran.

**Disclosures.**
- The `outer_test_rows_read_before_lock` field inside each H303-format lock is a hard-coded 0 from H303's code. H308's
  per-task guard receipts are the authoritative counters: built 7,314, accessed 0.
- The P0 rows come from H303's archived picks, so they inherit H303's old loader. P0 is reported for comparison only and
  does not enter the decision.

## Next step

1. **Use about +0.045 as the central estimate.** ENS_VOTE2 NESTED (H303 grid) is the preferred shadow selector at roughly
   +0.045 vs r882 slot0, with LCB95 > 0 across 3 partitions × 3 seeds.
2. **Keep the r3175 caveat.** The r3175-anchored effect remains unmeasured.
3. **Fresh confirmation needs data.** The real confirmation step needs a newly ingested S3 population (H299/H300); no
   further DL-Train resampling is needed.
4. **Reuse the loader.** Future DL-Train waves should use [trainonly_loader.py](trainonly_loader.py) and the guard
   pattern.

## Files and custody

| File | Contents |
|---|---|
| [PREREG.md](PREREG.md) | frozen protocol (commit `2e2a22eda5`) |
| [trainonly_loader.py](trainonly_loader.py) | TRAIN-prefix loader, packets guard |
| [run_h308.py](run_h308.py), [evaluate_h308.py](evaluate_h308.py), [test_h308.py](test_h308.py) | regression / fit / score with guards, evaluation and decision, 7 tests incl. the negative fixture |
| [metrics.json](metrics.json) | compact metrics: per-partition judge, concentration, correlations, histograms, lock hashes, loader and guard counters |
| [regression_receipt.json](regression_receipt.json), [s3_manifest.json](s3_manifest.json), [manifest.json](manifest.json) | regression, S3 VersionIds, pins |

**S3:** `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H308/` holds:
- `h308_models_locks_and_outputs.tar.gz` (VersionId `kA4PRbNsYKJTjAJIYLTbp0W58dX3qb2L`, sha256 `1304eae8…`, 534 MB);
- `metrics_full.json` (`QrVmJhid…`);
- `regression_receipt.json` (`L6SNMPoP…`);
- `picks.npz` (`uXsQkA27…`).

Every object passed an exact-version GET and sha256 match before local scratch was deleted.
