# H308: H303 NESTED ENS_VOTE2 through a TRAIN-only loader, on two new outer partitions

**Owner:** rd_opus [OPUS]. **Assignment:** `~/nudges/hw_t0735_opus_h308.md` (Grok Bot H-wave pass, 2026-11-04 07:35Z,
decided). **Claim:** `~/jope_plan/ns/receipts/H308_claim.txt`.

**Integrity.**
- CPU only (xbabe2) and DL-Train only (`selected.json` `981960ad…`, 1,219 contexts, 205 families).
- **No DL-Val feature row is decoded** (step 0 below).
- No neural forward or update; DOPE `c51a56d0` is untouched.
- Nothing on xbabe1 or xbabe3.
- Shadow research only. Neither verdict authorizes serving; fresh confirmation still needs a new S3 ingest (H299/H300).

**Publication.** This prereg (with the loader, tests, decision rule and partition salts) is pushed non-force to
`origin/v1/next` **before any new-partition fit or score**.

## Why

H303 is NESTED_CONFIRMED (+0.0643 [+0.0223, +0.1336] vs r882 slot0), and H305 reproduced it. H305 left two holes:
- **(h) Loader.** The H302/H303/H306 loaders decode the 91 DL-Val feature rows of the mixed H58 cache NPZs before TRAIN
  filtering, and their `outer_test_rows_read_before_lock=0` counters overstate isolation.
- **One partition.** Every ENS_VOTE2 result uses one outer partition (P0 = SHA256("H302-outer-v1" NUL family) mod 4),
  and seeds 30002/30003 are near-duplicates (family-gain corr .98). The gain may be partition luck.

H306 (XGRID_NOT_ADVANCED), H307 and H291 rule out re-extending grids or retrying harm-veto. H308 keeps everything frozen
and changes only the loader and the partition.

## Step 0: clean-room TRAIN-only loader ([trainonly_loader.py](trainonly_loader.py), committed with this prereg)

The loader handles `pool_priors.npz` and `context_features.npz` as follows:
1. It reads only the `cids` and `offsets` members, which hold identifiers and row offsets, not features.
2. It verifies that the first 1,219 cids are exactly the TRAIN cids and that no later cid is TRAIN. This was verified on
   the real files: 1,310 cids, with TRAIN as the prefix; pool rows `[0, 1404287)` of 1,509,119.
3. For each feature member, it reads **only the TRAIN prefix byte range**, with `os.pread` on the uncompressed `.npy`
   payload.

[run_h308.py](run_h308.py) installs it as `np` inside H303's unchanged `setup()`. The other npz files are unchanged:
`r882-train-costs.npz` is keyed per TRAIN cid.

**Negative fixture ([test_h308.py](test_h308.py), 7 tests).** NPZs whose trailing non-TRAIN rows are overwritten in
place with NaN:
- `np.load` (a whole-member read) raises a CRC `BadZipFile`;
- the loader returns exactly the TRAIN prefix, with no NaN;
- its byte counter equals the prefix size;
- a non-prefix layout is refused.

**Honest counters.** All 1,219 TRAIN packets (features and utilities) are built once, before the fork, so every
outer-test packet is **present in memory** before its task's fit_lock. That count is recorded per task. Inside each task,
the packets dict is replaced by a guard that **raises on any outer-test access** until the lock is written, so
**accessed = 0** is enforced, not asserted. Both counts go into `guard_receipt.json` per task.

## Step 1: regression first (with the new loader)

H303 outer fold 0 / seed 30001 (partition P0) is run with H303's grid and the new loader. It must reproduce the archived
H303 lock:
- the choice (TAU 0.05, k 5, a 2);
- inner gain 0.0961251;
- the 18-row inner table;
- every member model sha256;
- fold-0 outer-test picks with sha256 `18fdb2538ceabd9a9e5d07ac4a80cf9ff0eaf7597d9dc5fd1645acf0566bd3aa`.

The archive is the S3 H303 tar, VersionId `hshnLPUZuvpHewykQNTh..mWAGu9KGHo`, sha `8957cf0e…`. Any mismatch means **stop
and write `REGRESSION_FAIL`**. The fit phase refuses to run without a PASS receipt.

## Step 2: REPARTITION

**Frozen H303 NESTED**, imported with sha256 pins ([manifest.json](manifest.json)):
- H287 `213b30ad52` and `selected.json` `981960ad…`;
- the 18-config grid TAU {0.05, 0.10, 0.15} × k {5, 6, 7}/8 × a {2, 3}/3;
- inner 4-fold family CV inside outer_train (SHA256("H303-inner-v1" NUL str(fold) NUL family) mod 4);
- the 1e-4 conservative tie rule;
- root seeds 30001/30002/30003 and their member seeds;
- refit, fit_lock, and outer_test scored once.

**New outer partitions** (salts declared here):
- **P1:** SHA256("H308-outer-v1" NUL family), last byte mod 4.
- **P2:** SHA256("H308-outer-v2" NUL family), last byte mod 4.

That is 2 partitions × 4 folds × 3 seeds = 24 tasks. If time is tight, P2 is dropped and that is said in Deviations;
seeds are never dropped.

**P0** (H303's partition) is reported alongside from H303's archived picks, with no refit.

## Reporting

**Per (partition, seed), plus the median over seeds per partition:**
- family-equal gain vs r882 slot0 with a 10,000-draw whole-family bootstrap CI (RNG 302150), and LCB95;
- family-median gain;
- W/T/L, harmed share and invalid excess;
- drop-top-1/5/10 gains and drop-top-5 LCB95;
- top-10 share;
- the chosen-config histogram;
- per-family gain correlation across seeds.

**Elo:** MM Bradley–Terry, labelled exactly **"reference = r882 slot0 (DL-Train OOF); r3175-anchored Elo unmeasured"**.

No AUC, and no speed verdicts.

## Decision ([evaluate_h308.py](evaluate_h308.py); ties never win)

**PARTITION_ROBUST** iff all five hold:
1. On **every** new partition, the median gain over seeds is > +0.001 **and** the pooled one-sided LCB95 is > 0.
2. The per-seed one-sided LCB95 is > 0 in **≥ 5/6** (partition, seed) cells.
3. The drop-top-5 one-sided LCB95 is > 0 in **≥ 4/6** cells.
4. The harmed share is ≤ 35% in every cell.
5. The invalid excess vs slot 0 is ≤ 3 in every cell.

If P2 is dropped, parts 2 and 3 become ≥ 3/3 and ≥ 2/3 of P1's cells.

**Otherwise PARTITION_FRAGILE**, naming which criterion failed.

## Resources

- **Hosts:** xbabe2 CPU only, nice 19.
  - The regression runs on core 121.
  - The fit uses `taskset` 100, 101, 102, 112, 113, 115, 123 and 125 (87–96% idle in an 8 s sample; host 20% busy),
    with 8 forked 1-thread workers.
- **Memory:** systemd `MemoryMax=8G`; stop if MemAvailable < 20 GiB.
- **Scratch:** ≤ 5 GB under `~/jope-data/h308`, archived to S3 `…/H308/` (exact-version GET and sha verify), then deleted.
- **Deadlines:** compute stop 09:15Z; close 09:30Z.
