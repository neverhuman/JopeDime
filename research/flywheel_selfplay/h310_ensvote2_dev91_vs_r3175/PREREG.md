# H310: one-shot dev91 transfer test of frozen H303 NESTED ENS_VOTE2 vs the incumbent r3175

**Owner:** rd_opus [OPUS]. **Assignment:** `~/nudges/hw_t0827_opus_h310.md` (Grok Bot H-wave pass, 2026-11-04 08:27Z,
decided). **Claim:** `~/jope_plan/ns/receipts/H310_claim.txt`.

**Integrity.**
- CPU only (xbabe2), with no GPU and no neural forward or update; DOPE `c51a56d0` is untouched.
- No sealed, holdout, meta_test, fresh-cohort or DL-Test read.
- This is the **only** DL-Val use, and it happens **after** the DL-Train lock (Jepson's Oct 7 ruling: dev contexts are
  training data; only sealed meta-test is holdout).
- No verdict authorizes serving or touches the flywheel. Fresh confirmation still needs a new S3 ingest.

**Publication.** This prereg is pushed non-force to `origin/v1/next` **before any lock fit and before any dev91 read**.
The score phase refuses to run unless this commit is an ancestor of `origin/v1/next`.

## Why

H308 (PARTITION_ROBUST, `6826ee680c`): H303 NESTED ENS_VOTE2 gives about +0.045 vs r882 slot0 on DL-Train OOF, over 3
partitions × 3 seeds. The remaining gap: that is measured vs r882 on DL-Train, **not vs the real incumbent r3175 on
dev91 (Native91)**.

**Conservative for the challenger.** r3175 was itself selected on dev91 by the flywheel, so winner's curse favours the
incumbent in this comparison.

## Design ([run_h310.py](run_h310.py), [evaluate_h310.py](evaluate_h310.py))

### Lock protocol (DL-Train only)

**Step 0(c), train regression, first.** H303 P0 fold 0 / seed 30001 is run through H308's TRAIN-only loader. It must
reproduce the archived H303 lock (0.05, 5, 2; inner gain 0.0961251; inner table; member hashes) and the fold-0 picks sha
`18fdb2538cea…`. A mismatch means REGRESSION_FAIL and a stop.

**Step 1, lock.** For root seeds 30001/30002/30003, H303 NESTED is refit unchanged:
- the same 18-config grid TAU {0.05, 0.10, 0.15} × k {5, 6, 7} × a {2, 3};
- inner 4-fold family CV over **all 205 DL-Train families**, with inner fold = SHA256("H303-inner-v1" NUL "ALL" NUL
  family) mod 4;
- 1e-4 conservative ties;
- the same pins (H287 `213b30ad52`, `selected.json` `981960ad…`) and member seeds.

This gives one frozen chooser per seed, recorded in an fsynced `fit_lock.json` (chosen config, inner table, member model
sha256). A lock receipt records each lock's sha256.

**Dev91 isolation before the lock (tested in [test_h310.py](test_h310.py)).**
- An audit hook refuses any open of a dev91 pool file (any dev cid in the path), dev91 costs (`secondary-costs`,
  `r3175-gpu-costs`) or dev91 reference tables (`round-3175`, `best_dev_checkpoint`), and counts refusals.
- The TRAIN-only loader decodes no DL-Val cache row.
- The packet dict is checked to hold no dev cid.
- `dev91_rows_read_before_lock` must be 0.

**Disclosure.** Before this prereg, the four dev91 input files were byte-hashed with `sha256sum` to pin them. Their
contents were not parsed. The dev family count (19) was read from `selected.json` metadata, which the pinned loader
reads whole anyway. No dev91 utility, cost or feature value has been read.

### Score (after the lock and this prereg)

**Order of the dev91 regressions.** To honour "no dev91 read before the lock", regressions (a) and (b) run **after the
lock and before any challenger pick**. A mismatch means REGRESSION_FAIL, and no score is computed.
- **(a) r3175.** Slot 0 of r3175's GPU-exact dev91 top-8 pool must equal the stored `round-3175.json` dev_utility
  per-context on 91/91, with mean 1.1389572403748425. Costs come from S3 H281 attempt-001 `r3175-gpu-costs.npz`,
  VersionId `EgHvHGbSUy6RSXD60ZKiItTdX8HGMysg`, sha `d99363f6…`. This is the H281 A_EXACT path, so no GPU is needed.
- **(b) r882.** Slot 0 of r882's dev91 top-8 pool (`r882-secondary-costs.npz`) must equal r882's `best_dev_checkpoint`
  per-context on 91/91, with mean 1.1377941898164525.

**Arms.** Each seed's frozen chooser picks **once**:
- **PRIMARY:** among r3175's top-8 pool, compared with **r3175 slot 0**. This is the same transfer setting as H281/H278
  P2.
- **Secondary:** the same picks vs **r882 slot 0** (the Native91 per-context values).
- **Descriptive:**
  - the chooser on r882's dev91 top-8 pool, vs r3175 slot 0;
  - a random-valid-slot null on the r3175 pool (RNG 310150);
  - a clipped-at-−1 sensitivity of the primary.

**Utility scale.** Flywheel dev_utility: gate ? −fitness : −2.0, so failures score −2.0.

## Reporting

**Per seed, plus the median over seeds:**
- family-equal gain vs r3175 with a 10,000-draw whole-family bootstrap CI (RNG 302150), and LCB95;
- family-median gain;
- W/T/L, harmed share and invalid excess;
- drop-top-1/5 gains and the drop-top-5 LCB95;
- switches;
- the chosen config per seed.

**Elo:** MM Bradley–Terry, labelled exactly **"reference = r3175 slot0 (dev91)"**.

**Power note.** 91 contexts in **19 families**. The MDE is reported from the observed family dispersion. With 19
families, the CI is expected to be wide.

No AUC, and no speed verdicts.

## Decision (ties never win)

**DEV91_TRANSFERRED** iff all six hold:
1. The median over seeds of the family-equal gain vs r3175 is > +0.001.
2. The pooled one-sided LCB95 is > 0.
3. The per-seed one-sided LCB95 is > 0 on ≥ 2/3 seeds.
4. The median over seeds of the drop-top-5 gain is > 0.
5. The harmed share is ≤ 35% each seed.
6. The invalid excess vs r3175 slot 0 is ≤ 1 each seed.

**DEV91_UNDERPOWERED** iff not transferred, the pooled 95% CI half-width is > 0.03, and the median gain is > 0.

**Else DEV91_NOT_TRANSFERRED.**

## Resources

- **Hosts:** xbabe2 CPU only, nice 19.
  - The regression runs on core 121.
  - The lock uses 3 forked 1-thread workers on cores 95, 106, 117, 119, 120 and 121 (90–95% idle; rd_astra's h309 cores
    56, 57, 59, 64, 66 and 67 are excluded).
- **Memory:** systemd `MemoryMax=8G`; stop if MemAvailable < 20 GiB.
- **Scratch:** ≤ 4 GB, archived to S3 `…/H310/` (exact-version GET and sha verify), then deleted.
- **Deadlines:** compute stop 10:45Z; close 11:00Z.
