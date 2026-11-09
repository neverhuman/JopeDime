# H310: one-shot dev91 transfer test of frozen H303 NESTED ENS_VOTE2 vs the incumbent r3175

**Verdict: `DEV91_TRANSFERRED`.** This is shadow research only: no serving change, and the flywheel was not touched.
Fresh confirmation still needs a new S3 ingest.

**What was tested.** The H303 NESTED ENS_VOTE2 procedure was locked on DL-Train only, then applied **once** to dev91
(Native91). There it reranks r3175's own top-8 proposals.

**Result vs the incumbent r3175's pick (slot 0):**
- Family-equal gains: **+0.0114 / +0.0153 / +0.0274**.
- Median **+0.0153**, pooled 95% CI **[+0.0084, +0.0256]**.
- One-sided LCB95 > 0 on 3/3 seeds.
- No invalid picks; harmed ≤ 10.5%.
- Context-mean dev utility: 1.15091 / 1.15489 / 1.16752, against r3175's 1.13896.

**The gain is concentrated.** With 19 dev91 families, dropping the top 5 families leaves the gain positive on 2/3 seeds.
Dropping the top 10 makes it slightly negative on every seed.

**Conservative for the challenger.** r3175 was itself selected on dev91 by the flywheel (winner's curse), so this
comparison favours the incumbent.

**Elo: reference = r3175 slot0 (dev91).**

**Records.**
- **Prereg:** [PREREG.md](PREREG.md), commit `028d4bf57b3e9d73bd9a7ddca9552d0f117b3e2a`, pushed non-force to
  `origin/v1/next` at 08:36:24Z. That was before any lock fit (08:44:30Z) and any dev91 read (08:57:40Z). The score phase
  checks that commit is an ancestor of `origin/v1/next`.
- **Owner:** rd_opus [OPUS].
- **Assignment:** `~/nudges/hw_t0827_opus_h310.md`.

## Hypothesis

H308 (PARTITION_ROBUST): H303 NESTED ENS_VOTE2 gives about +0.045 vs r882 slot0 on DL-Train OOF. The remaining gap is
whether that transfers to the real incumbent, **r3175 on dev91**.

## Setup

### Step 0(c): train regression first ([regression_train_receipt.json](regression_train_receipt.json)): PASS

H303 P0 fold 0 / seed 30001 ran through H308's TRAIN-only loader with the dev guard active. It reproduced H303's lock
exactly: (0.05, 5, 2), inner gain 0.0961251, the inner table, member hashes, and fold-0 picks `18fdb2538cea…`.

### Step 1: the lock, DL-Train only ([lock_receipt.json](lock_receipt.json))

For each root seed, H303 NESTED was refit unchanged on all 205 DL-Train families:
- the 18-config grid and inner 4-fold family CV;
- 1e-4 conservative ties;
- H287 `213b30ad52` pins and member seeds.

This gives one frozen chooser per seed:

| seed | chosen (TAU, k, a) | inner gain | inner switches | fit_lock sha256 |
|---|---|---|---|---|
| 30001 | 0.10, 5, 2 | 0.05640 | 212 | `a06cb7f09a77…` |
| 30002 | 0.05, 5, 2 | 0.05496 | 310 | `d78922f4a23d…` |
| 30003 | 0.05, 5, 2 | 0.04473 | 341 | `40f02d99a0fd…` |

**`dev91_rows_read_before_lock = 0`**, backed by four checks:
- an audit hook refused any open of a dev91 pool, dev91 costs or dev91 reference table: 0 refusals needed;
- the TRAIN-only loader read 2,811,012 feature rows, which is exactly the TRAIN prefix, and 0 DL-Val rows;
- no dev cid was present among the packets;
- the guard is tested in [test_h310.py](test_h310.py).

The lock receipt sha256 is `bcf3e795…`, logged in the claim file before scoring.

### Step 0(a, b) and the score, after the lock

**Regressions.** Both run before any challenger pick:
- **(a) r3175.** Slot 0 of r3175's GPU-exact dev91 top-8 pool equals the stored `round-3175.json` dev_utility on
  **91/91**, mean **1.1389572403748425**. Costs come from H281 S3 `r3175-gpu-costs.npz`, VersionId `EgHvHGbS…`,
  sha `d99363f6…`.
- **(b) r882.** Slot 0 of r882's dev91 top-8 pool equals r882's best_dev_checkpoint on **91/91**, mean
  **1.1377941898164525**.

**One-shot score.** Each frozen chooser picked once on r3175's pool (primary) and once on r882's pool (descriptive).

**Run.** xbabe2 CPU, nice 19, cores 95/106/117/119/120/121, 3 workers, cgroup 8G (never hit; the watchdog never fired):
- train regression 08:34–08:44Z;
- lock 08:45:47–08:57:33Z;
- score 08:57:40–08:57:52Z.

## Outcome (dev91: 91 contexts, 19 families)

### Primary: frozen chooser on r3175's top-8 vs r3175 slot 0 (Elo: reference = r3175 slot0 (dev91))

| seed | family-equal gain [95% CI] | LCB95 | family median | W/T/L | sign p | harmed | invalid (r3175, arm) | switches | context mean (r3175 1.13896) | Elo [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|
| 30001 | +0.01145 [+0.00059, +0.02505] | +0.00196 | 0.0000 | 6/9/4 | 0.377 | 5.3% | 0, 0 | 17 | 1.15091 | 1570.3 [1341.3, 1880.2] |
| 30002 | +0.01527 [+0.00605, +0.02459] | +0.00755 | +0.0111 | 11/5/3 | 0.029 | 10.5% | 0, 0 | 34 | 1.15489 | 1725.3 [1526.7, 2745.7] |
| 30003 | +0.02736 [+0.01085, +0.04696] | +0.01298 | +0.0050 | 11/5/3 | 0.029 | 0.0% | 0, 0 | 25 | 1.16752 | 1725.3 [1543.6, 2731.8] |
| **median** | **+0.01527** [pooled **+0.00839, +0.02561**] | pooled +0.00968 | +0.0050 [0, +0.0259] | | | max 10.5% | excess 0 | | | **1725.3** [1526.7, 1930.1] |

The pick agreement between seeds is 69% / 86% / 69%. The clipped-at-−1 sensitivity is identical, because no utility is
below −1.

### Concentration (primary)

| seed | gain after dropping top 1 / 5 / 10 families | drop-top-5 LCB95 |
|---|---|---|
| 30001 | +0.0070 / −0.0011 / −0.0027 | −0.0031 |
| 30002 | +0.0135 / +0.0062 / −0.0037 | −0.0009 |
| 30003 | +0.0208 / +0.0062 / −0.0013 | +0.0007 |

The top-10 families carry 102–111% of the net gain; the other 9 families are net slightly negative.

### Secondary and descriptive arms

| arm | per-seed gain | median [pooled CI] | LCB95 > 0 | harmed | median Elo |
|---|---|---|---|---|---|
| same picks vs **r882** slot 0 (Native91) | +0.0126 / +0.0164 / +0.0285 | +0.0164 [+0.0092, +0.0270] | 3/3 | ≤ 10.5% | 1740.4 |
| chooser on **r882's** dev91 top-8 vs r3175 slot 0 | +0.0112 / +0.0132 / +0.0248 | +0.0132 [+0.0061, +0.0253] | 3/3 | ≤ 15.8% | 1620.2 |
| random valid slot on r3175's pool (RNG 310150) | +0.0037 / −0.0110 / −0.0239 | −0.0110 [−0.0493, +0.0294] | 0/3 | 32–42% | 1518.3 |

The oracle best slot in r3175's top-8 averages 1.27728, so headroom is about +0.138 per context.

### Power note

- 91 contexts in 19 families. The per-seed family-equal SE is 0.0065 / 0.0048 / 0.0096.
- The one-sided MDE (α 0.05, power 0.8) is about **0.016**, roughly the observed median. This test is barely powered.
- With 19 clusters, the whole-family bootstrap can understate uncertainty. The pooled CI half-width (0.0086) should be
  read as optimistic.

## Verdict

**Pre-registered rule.**
1. **PASS:** median +0.0153 > +0.001.
2. **PASS:** pooled LCB95 +0.0097 > 0.
3. **PASS:** per-seed LCB95 > 0 on 3/3 seeds.
4. **PASS:** median drop-top-5 gain +0.0062 > 0.
5. **PASS:** harmed ≤ 10.5%.
6. **PASS:** invalid excess 0.

**DEV91_TRANSFERRED.**

**Reading.**
- The frozen DL-Train-locked chooser improves on the flywheel incumbent's own dev91 picks on every seed, by about +0.015
  family-equal.
- That holds even though r3175 was selected on dev91 (a winner's-curse bias toward the incumbent).
- The effect is about a third of the DL-Train OOF effect vs r882 (+0.045). Moving from r882's pool to r3175's pool, and
  from DL-Train to dev91, shrinks it.
- It is sparse: a few of the 19 families carry it, and drop-top-10 is slightly negative on all seeds.
- This is a positive transfer signal, not a breakthrough claim. The sample is too small for that, and only a fresh
  population can confirm it.

## Deviations (declared)

1. **Regression order.** Regressions (a) and (b) ran after the lock and before any challenger pick (declared in the
   PREREG), to honour "no dev91 read before the lock".
2. **First lock launch crashed.** The first lock launch (08:44:30Z) crashed within 25 s, before any lock or inner score.
   H302's member fit casts `outer_fold` to `int`, so the string key "ALL" failed.
   - The key became int **4** (outside folds 0–3; it is only a filename/record label), so the inner fold salt is
     SHA256("H303-inner-v1" NUL **"4"** NUL family) instead of the PREREG's "ALL". It is equally pre-declared, arbitrary
     and outcome-blind.
   - Partial member files from the crash (no inner score) are archived in the S3 tar (`lock_attempt0_crashed_str_key/`).
   - `run_h310.py` and `evaluate_h310.py` changed in 4 lines in total; [manifest.json](manifest.json) records both
     versions.
3. **Pre-lock byte hashing (disclosed in the PREREG).** Before the prereg, the 4 dev91 input files were byte-hashed with
   `sha256sum` to pin them, without parsing. The dev family count (19) was read from `selected.json` metadata.

## Next step

1. **Shadow candidate, not serving.** ENS_VOTE2 NESTED (H303 grid, locked on DL-Train only) beats r3175 on dev91 with
   LCB > 0 on 3/3 training seeds. Compare with H281's frozen RERANK8_VETO P2 on the same r3175 pool. It is a candidate
   for the flywheel shadow observer: shadow logging only, with owner approval for any wiring.
2. **Fresh confirmation is required.** The decisive test is a newly ingested S3 population (H299/H300), scored once
   against r3175 with this lock. Dev91 (19 families, concentrated gain, MDE ≈ observed) cannot carry a breakthrough
   claim alone.
3. **Frozen artefacts.** Use the 3 locks archived in S3 `H310/`; do not refit.

## Files and custody

| File | Contents |
|---|---|
| [PREREG.md](PREREG.md) | frozen protocol (commit `028d4bf57b`) |
| [run_h310.py](run_h310.py) | regress-train / lock (dev guard) / score phases |
| [evaluate_h310.py](evaluate_h310.py) | regressions (a, b), one-shot score, arms, decision |
| [test_h310.py](test_h310.py) | dev-guard, decision and power tests |
| [metrics.json](metrics.json) | all arms, concentration, power note, regressions, lock summary |
| [lock_receipt.json](lock_receipt.json), [regression_train_receipt.json](regression_train_receipt.json), [s3_manifest.json](s3_manifest.json), [manifest.json](manifest.json) | receipts, S3 VersionIds, pins |

**S3:** `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H310/` holds:
- `h310_locks_models_outputs.tar.gz` (VersionId `1JWgsGRcV3fD5r4hvAJXptCQ_crMYoQl`, sha256 `f42107b6…`, 85 MB), with the
  3 frozen locks and members;
- `metrics_full.json`, `picks.npz`, `lock_receipt.json` and `regression_train_receipt.json`.

Every object passed an exact-version GET and sha256 match before local scratch was deleted.
