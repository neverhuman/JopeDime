# H415: a committee of trained screeners and a 5-fold cross-fit bag over the population (TRAIN P201e)

**Verdict: `KFB_CONFIRMED`, with the gain carried by S2.** This is a TRAIN-only result. dev91 was not read.

- **Primary passes.** `P3+LIB3.KFB` against the locked baseline X0e has gain +0.131, a **12.3% RMSE cut**.
  - Families W/T/L: 126/1/58.
  - Clipped mean +0.057, sign-flip p 2.2e-5, LOFO +0.054, bootstrap LCB +0.030.
- **S1 fails.** KFB against H410's best builder `P3+LIB3.BAG_CAR` has gain −0.002 (p 0.26). Cross-fitted selection and
  bagging is no better than H410's builder on the same pool.
- **S2 passes.** `P2+LIB2.BAG_CAR` against X0e has gain +0.130, a 12.2% RMSE cut (p 7.1e-4). This is a fresh,
  independent replication of H410's pooled-runs-plus-library TRAIN gain (H410: +8.1%).
- **S3 fails.** The scale-L screener alone in the loop (`GL` stock) has gain +0.005 (p 0.28).
- **TRAIN Elo** (anchor X0e = 1500; 185 contexts):
  - `P3+LIB3.HFR` 1655.3 [1606.9, 1705.8];
  - `P3+LIB3.KFB` 1645.3 [1597.8, 1693.3];
  - `P3+LIB3.W48` 1630.5.

**What this means.** Pooling the programs of several runs that used different screeners, and adding the strong-learner
library, is now a replicated TRAIN gain of about 12% RMSE on a fresh panel. The two new parts add nothing measurable on
top:
- the trained screener committee (S3);
- 5-fold cross-fit selection and bagging (S1).

KFB and HFR are the strongest TRAIN outputs and are statistically level with the H410-style `BAG_CAR` and `W48`
builders.

- **PREREG:** [`PREREG.md`](PREREG.md):
  - part A at `3293d21144`;
  - amendment 1, L-screener fold pinning, at `a36054add0`;
  - amendment 2, the deploy code and DEV2E smoke, at `26a31b8403`.
  All three were published before the first deploy output.
- **Owner:** `cl_fwriter`, Claude Opus 5.5. Claim: `ns/receipts/H415_claim.txt`.
- **Machine-readable results:** [`metrics.json`](metrics.json), built from `results/H415_DEPLOY_P201E.json` (in the
  closeout bundle) without re-estimation.
- **Archive:** [`ARCHIVE.json`](ARCHIVE.json). It is a create-only S3 bundle, verified by an exact-version GET. 925 of
  its 926 members match their recorded hashes. The one mismatch is the upload's own process log, which was still being
  written while the bundle was made. It holds:
  - the H415-L screener (10 fold × seed models, TRAIN only);
  - the run and stage records;
  - the result files;
  - the evidence directory.

## Panel and runs
- **P201e:** the fifth seeded context of every TRAIN family with at least 5 contexts (PREREG §3). That is 185 contexts in 185
  families.
- **Three locked H335 AGG searches per context,** one per screener:
  - `GS` with `H0.1-A`;
  - `GE` with the grown S3 `H381-E2-S3fz`;
  - `GL` with the new scale-L `H415-L`.
- **Stock replays are exact.** Max |Δ| = 0 on 185/185 contexts for GS, GE and GL, and the members are identical.
- **KFB pool:**
  - all 185 envelopes are usable;
  - 63.8 usable candidates per context on average;
  - Caruana members average 6.6;
  - the selected ensemble beats the best single program's out-of-fold u by 0.053 on average.
- **Worker time** (worker-seconds): search GS 191,891, GE 185,836, GL 238,437; library 103,159; HF re-scoring
  201,251; KFB deploy 17,494; KFB1 2,870.

## Integrity
- **Training data only.** No dev91, DL-Val, DL-Test or sealed byte was read, and no holdout model was rescored.
- **No stored baseline was rerun.** X0e is the stored locked recipe, replayed exactly.
- **DOPE** `c51a56d0` is unchanged.
- **Process note.** The deploy parent ran at nice 0 inside docker: the systemd unit's `Nice` setting did not reach the
  container. The results are unaffected; the gap is in the run wrapper.

## Next step (owner decision; not run in H415)
A confirmed KFB, or the simpler `P3+LIB3.BAG_CAR`, which S1 cannot separate from it, is the candidate for **one** locked
dev91 once-score under its own PREREG, LOCK and receipt (H369 pattern).
